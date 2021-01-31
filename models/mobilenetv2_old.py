'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MODULE.dynamic_conv import Dynamic_conv2d
from.MODULE.dynamic_depthwise_conv import DynamicGroup_conv2d
from .MODULE.dydconv import DYDConv2d

def DepthwiseConv2d(planes, kernel_size=3, stride=1, 
                    padding=1, bias=False, dynamic='conv2d', K=4, temperature=30):
    if dynamic =='dynamic':
        return Dynamic_conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                           groups=planes, bias=False, K=K, temperature=temperature)
    elif dynamic == 'dynamic_group':
        return DynamicGroup_conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                           groups=planes, bias=False, K=K, temperature=temperature)
    elif dynamic == 'dydconv':
        return DYDConv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                         groups=planes, bias=False, K=K, temperature=temperature, optim_forward=False)
    else:        
        return nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=planes, bias=False)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    
    def __init__(self, in_planes, out_planes, expansion, stride, dynamic):
        super(Block, self).__init__()
        self.stride = stride
        
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DepthwiseConv2d(planes, kernel_size=3, stride=stride, padding=1, bias=False, dynamic=dynamic)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    
    def __init__(self, dynamic='conv2d', num_classes=10):
        super(MobileNetV2, self).__init__()
        #self.dynamic = dynamic
        img_size, in_channels, out_channels = 32, 320, 1280
        # (expansion, out_planes, num_blocks, stride)    
        self.cfg = [(1,  16, 1, 1, None),
                    (6,  24, 2, 2, None),  # NOTE: change stride 2 -> 1 for CIFAR10
                    (6,  32, 3, 2, None),
                    (6,  64, 4, 2, dynamic),
                    (6,  96, 3, 1, dynamic),
                    (6, 160, 3, 2, dynamic),
                    (6, 320, 1, 1, dynamic)]
        
        self.conv1 = nn.Conv2d(3, img_size, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(img_size)
        self.layers = self._make_layers(in_planes=img_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(out_channels, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride, dynamic in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, dynamic))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

