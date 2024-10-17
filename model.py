from utils import *
import torch
from torch import nn
import torchvision
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


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
        self.dil_conv2 = nn.Conv2d(outplanes,outplanes,kernel_size=kernel_size,dilation=dilation)
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
    
inplanes = 3
outplanes = 6
expansion = 1
kernel_size = 2
stride = 2
dilation = 2
downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*expansion, kernel_size=5, stride=stride),
                nn.BatchNorm2d(outplanes * expansion),
            )
conv = MSFE(inplanes,outplanes,kernel_size,stride,downsample,dilation)
input = torch.randn(20, 3, 224, 224)
output = conv(input)

print(output.shape)