import torch
import torch.nn as nn
import torch.nn.functional as F


class CWFF(nn.Module): # cognitive weights from features
    def __init__(self, in_planes, out_planes, ratios=1.):
        super(CWFF, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)
    
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 1, bias=True)
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
        

class COGConv2d(nn.Module): # Cognitive convolution
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=None, in_chunk=4):
        super(COGConv2d, self).__init__()
        assert bias is None # no bias
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.in_chunk = in_chunk
        
        self.cwff = CWFF(in_planes, (in_chunk * in_planes) //self.groups, ratios=1.)
        
        gain = nn.init.calculate_gain('relu')
        cog_he_std = gain * (in_chunk * kernel_size ** 2) ** (-0.5)  # He init
        he_std = gain * (in_planes * kernel_size ** 2) ** (-0.5)  # He init

        self.cog_weight = nn.Parameter(torch.randn(out_planes, in_chunk, kernel_size, kernel_size) * cog_he_std, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes //self.groups, kernel_size, kernel_size) * he_std, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def __cognitive_weights_conv(self, B, kernel):
        """
            B : batch size
            kernel : B * (in_chunk * in_planes) * 1 * 1
        """
        kernel = kernel.view(-1, self.in_chunk, 1, 1) 
        cognitive_weights = F.conv2d(self.cog_weight, weight=kernel, bias=None, stride=1, padding=0) 
        cognitive_weights = cognitive_weights.view(self.out_planes, B, -1).permute(1,0,2).contiguous()
        weight = self.weight.view(1, self.out_planes, -1)

        cognitive_weights = self.sigmoid(cognitive_weights) * weight
        cognitive_weights = cognitive_weights.view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        return cognitive_weights

    def forward(self, x):
        B, _, H, W = x.size()
        kernel = self.cwff(x)
        cognitive_weights = self.__cognitive_weights_conv(B, kernel)
        x = x.view(1, -1, H, W)

        output = F.conv2d(x, weight=cognitive_weights, bias=self.bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*B)
        output = output.view(B, self.out_planes, output.size(-2), output.size(-1))                              
    
        return output


if __name__ == "__main__":
    x = torch.randn(64, 64, 224, 224)
    module = COGConv2d(64, 64, kernel_size=3, padding=1, groups=64)
    out = module(x)
    print(out.size())
    
    
    
