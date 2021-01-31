import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

#import dyd_C

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

def check_options(in_channels, out_channels, bias):
    if in_channels != out_channels:
        raise ValueError('DYDConv2d does not support in_channels != out_channels')

    if bias:
        raise ValueError('DYDConv2d does not support bias')


class DYDConv2d_OF(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, pad):
        ctx.stride = stride
        ctx.pad = pad
        
        ctx.save_for_backward(input, weight)

        _, _, in_h, in_w = input.size()
        out = dyd_C.dydconv2d(
                input, weight, 1, 1, *stride, pad[0], pad[0], pad[1], pad[1], True
        )

        _, _, out_h, out_w = out.size()

        ctx.g_pad = (
            weight.size(2) - pad[0] - 1,
            in_h - out_h * stride[0] + pad[0],
            weight.size(3) - pad[1] - 1,
            in_w - out_w * stride[1] + pad[1],
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        stride = ctx.stride
        pad = ctx.pad
        g_pad = ctx.g_pad

        return None, None, None, None, None, None


class DYDConv2d_O(nn.Module): #dynamic depthwise conv2d optim version
    def __init__(self, in_channels, out_channels, kernel_size, weight, stride=1, 
                 padding=0, bias=False):
        super().__init__()

        check_options(in_channels, out_channels, bias)

        self.stride = make_tuple(stride, 2)
        self.padding = make_tuple(padding, 2)
        self.in_channel = in_channels
        self.kernel_size = kernel_size
        self.weight = weight
        
    def forward(self, input):
        return DYDConv2d_OF.apply(input, self.weight, self.stride, self.padding)
