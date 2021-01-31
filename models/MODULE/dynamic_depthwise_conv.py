import torch
import torch.nn as nn
import torch.nn.functional as F


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, CK, temperature, init_weight=True):
        super(attention2d, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = CK
    
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, CK, 1, bias=True)
        if init_weight:
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
        return x
        


class DynamicGroup_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4,temperature=30, init_weight=True):
        super(DynamicGroup_conv2d, self).__init__()
        assert in_planes%groups==0
        assert bias==False

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        self.attention = attention2d(in_planes, ratio, K*out_planes, temperature)
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size, K), requires_grad=True)
        
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[...,i])

    def forward(self, x):
        attnetion = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        softmax_attnetion = F.softmax(attnetion.view(-1, self.K, self.out_planes), dim=1)
        print(softmax_attnetion.size())
        softmax_attnetion = softmax_attnetion.view(-1, self.K*self.out_planes, 1, 1)
        print(softmax_attnetion.size())
        weight = self.weight.view(-1, self.K, 1, 1)
        print(weight.size())
        weight = F.conv2d(softmax_attnetion, weight=weight, groups=self.out_planes)
        print(weight.size())
        x = x.view(1, -1, height, width)
        
        aggregate_weight = weight.view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))                              
        return output        


if __name__ == "__main__":
    x = torch.randn(64, 64, 224, 224)
    model = DynamicGroup_conv2d(in_planes=64, out_planes=64, groups=64, kernel_size=3, ratio=0.25, padding=1, bias=False)
    x = x.to('cuda:3')
    model.to('cuda:3')
    model(x)
    #print(model(x).size())

  