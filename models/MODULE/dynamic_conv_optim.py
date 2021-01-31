import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(SEModule, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
        self.hidden_planes = hidden_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
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
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, dim=1)
        #return F.softmax(x / 8, dim=1)


class DyConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, 
                 K=4,temperature=30, inference=False):
        super(DyConv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.inference = inference
        self.plotting = True
        self.plot_attention_values = None

        self.se_attention = SEModule(in_planes, ratio, K, temperature)

        gain = nn.init.calculate_gain('relu')
        he_std = gain * (in_planes * kernel_size ** 2) ** (-0.5)  # He init

        self.weight = nn.Parameter(torch.randn(K * out_planes, in_planes//groups, 
                                   kernel_size, kernel_size) * he_std, requires_grad=True)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K * out_planes))
        else:
            self.bias = None
        
    def forward_infer(self, x):
        attention = self.se_attention(x)
        
        B, _, H, W = x.size()
        x = x.view(1, -1, H, W)

        weight = self.weight.view(self.K, -1)
        
        aggregate_weight = torch.mm(attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        
        if self.bias is not None:
            aggregate_bias = torch.mm(attention, self.bias.view(self.K, self.out_planes)).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*B)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * B)
            
        output = output.view(B, self.out_planes, output.size(-2), output.size(-1))
        return output        
        

    def forward(self, x):
        if self.inference:
            return self.forward_infer(x)
    
        B, _, H, W = x.size()
        attention = self.se_attention(x)

        if self.plotting:
            self.plot_attention_values = attention

        if self.groups == 1:
            out = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            x = torch.cat([x] * self.K, dim=1)
            out = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups * self.K)

        attention = attention.view(B, 1, self.K)
        output = out.view(B, self.K, -1)
        output = torch.bmm(attention, output).view(B, self.out_planes, out.size(-2), out.size(-1))
        return output

    
if __name__ == "__main__":
    x = torch.randn(64, 64, 224, 224)
    module = DyConv2d(in_planes=64, out_planes=64, kernel_size=3, ratio=0.25, groups=1, padding=1, bias=False)
    module.inference=True
    out1 = module(x)
    module.inference=False
    out2 = module(x)
    
    print(module.weight.size())
    print(out1[5][5][5][2:12])
    print(out2[5][5][5][2:12])
