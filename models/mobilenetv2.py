import torch
import torch.nn as nn
import math
#from .MODULE.dy_channel_wise_conv import DyC_Conv2d
#from .MODULE.dy_in_channel_wise_conv import DyC_Conv2d
from .MODULE.dy_channel_wise_conv_optim import DyC_Conv2d
#from .MODULE.dynamic_conv import Dynamic_conv2d
#from .MODULE.dynamic_conv_optim import DyConv2d
from .MODULE.dynamic_conv_optim_and_l import DyConv2d
from .MODULE.cognitive_conv import COGConv2d

def custom_conv(inp, oup, kernel_size, stride=1, padding=0, groups=1):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)
    #conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)
    #conv = DyC_Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)     
    #conv = COGConv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=None)
    #conv = DyConv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, inference=False)
    return conv

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        custom_conv(inp, oup, 1, 1, 0),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                custom_conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                custom_conv(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                custom_conv(inp, hidden_dim, 1, 1, 0),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                custom_conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                custom_conv(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        in_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2], 
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        if input_size == 32: # NOTE: change stride 2 -> 1 for CIFAR10, CIFAR100
            interverted_residual_setting[1][3] = 1 
        
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, in_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            out_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(in_channel, out_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(in_channel, out_channel, 1, expand_ratio=t))
                in_channel = out_channel
        # building last several layers
        self.features.append(conv_1x1_bn(in_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        #self.dropout = nn.Dropout(0.2)
        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()
        
    def inference_mode(self):
        for module in self.features.modules():
            if module.__class__.__name__ == 'DyConv2d':
                module.inference = True
            elif module.__class__.__name__ == 'DyC_Conv2d':
                module.inference = False       

    def training_mode(self):
        for module in self.features.modules():
            if module.__class__.__name__ == 'DyConv2d':
                module.inference = False                
            elif module.__class__.__name__ == 'DyC_Conv2d':
                module.inference = False                

    def attention_plotting(self):
        plot_attention_values = []
        for module in self.features.modules():
            if module.__class__.__name__ == 'DyConv2d':
                plot_attention_values.append(module.plot_attention_values.unsqueeze(2))
        plot_attention_values = torch.cat(plot_attention_values, dim=2)
        return plot_attention_values

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        #x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(num_classes, input_size, width_mult, pretrained=False, model_path=None):
    model = MobileNetV2(num_classes=num_classes, input_size=input_size, width_mult=width_mult)
    if pretrained:
        assert model_path is not None
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    net = MobileNetV2(num_classes=1000, input_size=224)
    
