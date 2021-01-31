'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MODULE.dynamic_conv import Dynamic_conv2d

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, groups, K, temperature):
        super(Block, self).__init__()
        self.stride = stride
        
        planes = expansion * in_planes
        self.conv1 = Dynamic_conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, groups=groups, bias=False, 
                                    K=K, temperature=temperature)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Dynamic_conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, 
                               groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Dynamic_conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, groups=groups, bias=False,  
                                    K=K, temperature=temperature)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                Dynamic_conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=groups, bias=False, 
                               K=K, temperature=temperature),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class DyMobileNetV2(nn.Module):
    
    def __init__(self, K, temperature, num_classes=10):
        super(DyMobileNetV2, self).__init__()
        self.K = K
        self.temperature = temperature
        img_size, in_channels, out_channels, = 32, 320, 1280
        # (expansion, out_planes, num_blocks, stride, groups)    
        self.cfg = [(1,  16, 1, 1, 1),
                    (6,  24, 2, 1, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                    (6,  32, 3, 2, 1),
                    (6,  64, 4, 2, 1),
                    (6,  96, 3, 1, 1),
                    (6, 160, 3, 2, 1),
                    (6, 320, 1, 1, 1)]
        
        self.conv1 = Dynamic_conv2d(3, img_size, kernel_size=3, stride=1, padding=1, groups=1, bias=False, 
                                    K=K, temperature=temperature)
        self.bn1 = nn.BatchNorm2d(img_size)
        self.layers = self._make_layers(in_planes=img_size)
        self.conv2 = Dynamic_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False, 
                     K=K, temperature=temperature)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(out_channels, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride, groups in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, groups, 
                              K=self.K, temperature=self.temperature))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out