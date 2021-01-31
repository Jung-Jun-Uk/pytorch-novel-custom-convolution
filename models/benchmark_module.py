import torch
import torch.nn as nn

import argparse
import math
import time
import sys
import torch
import os

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

#from models.mobilenetv2 import DepthwiseConv2d
from utils.general import select_device

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-f', '--features', type=int, default=32)
    parser.add_argument('-s', '--state-size', type=int, default=128)
    parser.add_argument('-r', '--runs', type=int, default=100)
    #parser.add_argument('--module', choices=['conv2d', 'dynamic', 'dydconv'])
    parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parser()
    device = select_device(opt.device, batch_size=opt.batch_size)

    X = torch.randn(opt.batch_size, 128, opt.features, opt.features).to(device)
    #module = DepthwiseConv2d(128, kernel_size=3, stride=1, 
    #                padding=1, bias=False, dynamic=opt.module, K=4, temperature=30).to(device)
    #module.forward_optim = True
    module2 = nn.Conv2d(128*4*4, 128*4*4, kernel_size=3, stride=1, padding=1).to(device)
    X2 = torch.randn(opt.batch_size, 128*4*4, 4, 4).to(device)
    forward_min = math.inf
    forward_time = 0

    for _ in range(opt.runs):

        start = time.time()
        y = module2(X2)
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

    scale = TIME_SCALES[opt.scale]
    forward_min *= scale    
    forward_average = forward_time / opt.runs * scale
    
    print('Forward: {0:.3f}/{1:.3f} {2}'.format(
        forward_min, forward_average, opt.scale))

    