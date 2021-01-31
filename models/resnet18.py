import torch
import torch.nn as nn
import torch.nn.functional as F

#from .MODULE.dy_channel_wise_conv import DyC_Conv2d
#from .MODULE.dy_in_channel_wise_conv import DyC_Conv2d
from .MODULE.dy_channel_wise_conv_optim import DyC_Conv2d
#from .MODULE.dynamic_conv import Dynamic_conv2d
from .MODULE.dynamic_conv_optim import DyConv2d
from .MODULE.cognitive_conv import COGConv2d

def custom_conv(inp, oup, kernel_size, stride=1, padding=0, groups=1, bias=False):
    #conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=bias)
    #conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)
    #conv = DyC_Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)     
    conv = COGConv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=None)
    #conv = DyConv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, inference=False)
    return conv


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, p=0.2):
        super(Bottleneck, self).__init__()
        self.conv1 = custom_conv(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = custom_conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = custom_conv(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.p = p
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                custom_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.s = stride
        self.conv1 = custom_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = custom_conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                custom_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        #out = self.bn1(self.conv1(x))
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        y = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        out = self.linear(y)
        return y

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

if __name__ == "__main__":
    print(ResNet18())