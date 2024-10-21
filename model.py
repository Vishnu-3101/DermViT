from utils import *
import torch
from torch import nn
import torchvision
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from  torchvision.ops.deform_conv import DeformConv2d


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
        print(out.shape)
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
        print(offset.shape)
        out = self.depthwise1(x)
        out = self.pointwise1(out)
        print(out.shape)
        out = self.deform_conv(out,offset)
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        return out


inplanes = 6
outplanes = 12
expansion = 1
kernel_size = 2
stride = 2
dilation = 3
downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*expansion, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(outplanes * expansion),
            )
conv = AFE(inplanes,outplanes,kernel_size,stride)
input = torch.randn(32, 6, 112, 112)
output = conv(input)

print(output.shape)