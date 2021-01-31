import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('./')
from models.MODULE.DYD_CONV_CUDA.dydconv_optim import DYDConv2d_OF
from utils.general import select_device


def make_tuple(value, n_value):
    if not isinstance(value, (list, tuple)):
        return (value,) * n_value

    else:
        n_item = len(value)

        if n_item > n_value:
            raise ValueError(
                f'Number items does not match with requirements: {n_item}, expected: {n_value}'
            )

        if len(value) == n_value:
            return value

        return value * n_value


class SEModule(nn.Module):
    def __init__(self, in_planes, ratios, CK):
        super(SEModule, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
    
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, CK, 1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class InvertedEx(nn.Module):
    def __init__(self, in_planes, ratios, C, K):
        super(InvertedEx, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #hidden_planes = int(in_planes*ratios)+1
        hidden_planes = int(in_planes*ratios)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, C*K, kernel_size=1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        

class DyC_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False, 
                 ratio=0.25, K=4,temperature=30):
        super(DyC_Conv2d, self).__init__()
        assert in_channels%groups==0
        assert bias==False
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.temperature = temperature

        gain = nn.init.calculate_gain('relu')
        he_std = gain * (in_channels * kernel_size ** 2) ** (-0.5)  # He init
        
        #self.se_module = SEModule(in_channels, ratio, out_channels*K)
        ratio=0.25
        self.hidden_planes = int(in_channels*ratio)
        self.se_module = InvertedEx(in_channels, ratio, out_channels, K)
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels // groups, kernel_size, kernel_size) * he_std
        )
    
    def forward(self, x):
        B, C, H, W = x.size()
        y = self.se_module(x)
        y = y.view(y.size(0), self.K, self.out_channels)
        #prob = F.softmax(y/self.temperature, dim=1).unsqueeze(3)    
        prob = F.softmax(y/(self.hidden_planes ** 0.5), dim=1).unsqueeze(3)    
        weight = self.weight.view(self.K, self.out_channels, -1).unsqueeze(0)

        aggregate_weight = (prob * weight).sum(dim=1)

        aggregate_weight = aggregate_weight.view(-1, self.in_channels//self.groups, self.kernel_size, self.kernel_size)
        x = x.view(1, -1, H, W)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * B)
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        return output


if __name__ == "__main__":
    x = torch.randn(64,64,224,224)
    module = DyC_Conv2d(in_channels=64, out_channels=128, groups=1, kernel_size=3, padding=1, bias=False)
    device = select_device('5', batch_size=64)
    print(device)
    x = x.to(device)
    module = module.to(device)
    
    out1 = module(x)
    print(out1.size())
