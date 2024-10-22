from utils import *
import torch
from torch import nn
import torchvision
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from  torchvision.ops.deform_conv import DeformConv2d
from functools import partial
from utils import _make_divisible, merge_pre_bn
from timm.models.layers import DropPath
from einops import rearrange

NORM_EPS = 1e-5

class MSFE(nn.Module):
    def __init__(
            self,
            inplanes:int,
            outplanes:int,
            kernel_size:int=1,
            stride:int=1,
            downsample : Optional[nn.Module]=None,
            dilation: int=1,
            norm_layer: Optional[Callable[...,nn.Module]] = None 
            #Can take any number and type of arguments and return type is nn.Module
    )->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes,outplanes,kernel_size=kernel_size,stride=stride)
        self.bn1 = norm_layer(outplanes)
        self.dil_conv2 = nn.Conv2d(outplanes,outplanes,kernel_size=1,dilation=dilation)
        self.bn2 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x:Tensor)->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dil_conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

class AFE(nn.Module):
    def __init__(
            self,
            inplanes:int,
            outplanes:int,
            kernel_size:int=1,
            stride:int=1,
            downsample: Optional[nn.Module]=None,
    )->None:
        super().__init__()
        self.depthwise1 = nn.Conv2d(in_channels=inplanes,out_channels = inplanes, kernel_size = 1,stride=1)
        self.pointwise1 = nn.Conv2d(in_channels=inplanes,out_channels = inplanes, kernel_size = 1)
        self.deform_conv = DeformConv2d(in_channels = inplanes, out_channels =inplanes, kernel_size=1,stride = 1)
        self.depthwise2 = nn.Conv2d(in_channels=inplanes,out_channels = inplanes, kernel_size = kernel_size,stride=stride)
        self.pointwise2 = nn.Conv2d(in_channels=inplanes,out_channels = outplanes, kernel_size = 1)
        self.downsample = downsample
    
    def forward(self,x):
        # identity = x
        offset = x

        out = self.depthwise1(x)
        out = self.pointwise1(out)

        out = self.deform_conv(out,offset)
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        return out

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, outplanes=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.outplanes = outplanes if outplanes is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.outplanes)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class HFE(nn.Module):
    def __init__(
            self,inplanes,outplanes
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes,outplanes,stride=1,kernel_size=1)
        self.group = nn.Conv2d(inplanes,outplanes,stride=1,kernel_size=1,groups=4)
        self.conv2 = nn.Conv2d(inplanes,outplanes,stride=1,kernel_size=1)

    def forward(self,x):

        out = self.conv1(x)
        out = self.group(out)
        out = self.conv2(x)

        return out

class EMFGCB(nn.Module):
    def __init__(
        self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
        mlp_ratio=2, head_dim=8, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        # self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        # self.mhca_out_channels = out_channels - self.mhsa_out_channels
    
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.norm1 = norm_func(out_channels)
        self.linformer = LinformerSelfAttention(out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.linformer_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.hfe_block = HFE(in_channels,out_channels)
        self.hfe_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels*2)
        self.mlp = Mlp(out_channels*2, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)


    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self,x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        # if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
        #     out = self.norm1(x)
        # else:
        out = x
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.linformer_path_dropout(self.linformer(out))
        out =  rearrange(out, "b (h w) c -> b c h w", h=H)
        x = x + out
        print(out.shape)

        out = out + self.hfe_path_dropout(self.hfe_block(out))
        x = torch.cat([x, out], dim=1)
        out = self.norm2(x)
        x = x + self.mlp_path_dropout(self.mlp(out))
        
        return x

class DermMultiNetBlock(nn.Module):
    def __init__(self,inplanes, outplanes, path_dropout, kernel_size = 1, stride=1):
        super().__init__()
        
        downsample1 = nn.Sequential(
                nn.Conv2d(inplanes, inplanes*2, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(inplanes*2),
            )
        downsample2 = nn.Sequential(
                nn.Conv2d(inplanes*2, inplanes*4, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(inplanes*4),
            )
        self.msfe = MSFE(inplanes=inplanes,outplanes=inplanes*2,kernel_size=2,
            stride=2,
            downsample = downsample1,
            dilation = 3,
        )
        self.afe = AFE(inplanes*2,
            inplanes*4,
            kernel_size=2,
            stride=2,
            downsample = downsample2)
        self.emfgcb = EMFGCB(inplanes*4,outplanes,path_dropout)

    def forward(self,x):
        out = self.msfe(x)
        print(out.shape)
        out = self.afe(out)
        out = self.emfgcb(out)
        return out

class DermMultiNet(nn.Module):
    def __init__(self,in_dim,out_dim,path_dropout,kernel_size,stride):
        super().__init__()
        self.derm_block1 = DermMultiNetBlock(in_dim[0],out_dim[0],path_dropout,kernel_size,stride)
        self.derm_block2 = DermMultiNetBlock(in_dim[1],out_dim[1],path_dropout,kernel_size,stride)
        self.derm_block3 = DermMultiNetBlock(in_dim[2],out_dim[2],path_dropout,kernel_size,stride)
    
    def forward(self,x):
        out = self.derm_block1(x)
        print(out.shape)
        out = self.derm_block2(out)
        out = self.derm_block3(out)
        return out

in_dim = [3,24,192]
out_dim= [24,192,1536]
stride = 2
kernel_size = 2
path_dropout = 0.01



# NUM_CLASSES = 2
# PATCH_SIZE = 4
# IMG_SIZE = 56
# IN_CHANNELS = 12
# NUM_HEADS = 8
# DROPOUT = 0.001

# EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS # 16
# NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 # 49

# inplanes = 6
# outplanes = 12
# expansion = 1
# kernel_size = 2
# stride = 2
# dilation = 3
# downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, outplanes*expansion, kernel_size=kernel_size, stride=stride),
#                 nn.BatchNorm2d(outplanes * expansion),
#             )
conv = DermMultiNet(in_dim,out_dim,path_dropout,kernel_size,stride)
input = torch.randn(32, 3, 224, 224)
output = conv(input)

print(output.shape)