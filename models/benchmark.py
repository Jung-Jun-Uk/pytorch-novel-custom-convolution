import argparse
import math
import time
import sys
import torch
import os

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from models.build import build_models
from utils.general import select_device

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-f', '--features', type=int, default=32)
    parser.add_argument('-s', '--state-size', type=int, default=128)
    parser.add_argument('-r', '--runs', type=int, default=200)
    parser.add_argument('--model', choices=['mobilenetv2', 'mobilenetv2_dy_depth', 'mobilenetv2_dy_depth_group', 'dy_mobilenetv2', 'dy_ensemble', 'mobilenetv2_dyd'])
    parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parser()
    device = select_device(opt.device, batch_size=opt.batch_size)

    X = torch.randn(opt.batch_size, 3, opt.features, opt.features).to(device)
    model = build_models(opt.model, 10).to(device)
    #model.forward_optim = True
    forward_min = math.inf
    forward_time = 0

    for _ in range(opt.runs):

        start = time.time()
        y = model(X)
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

    scale = TIME_SCALES[opt.scale]
    forward_min *= scale    
    forward_average = forward_time / opt.runs * scale
    
    print('Forward: {0:.3f}/{1:.3f} {2}'.format(
        forward_min, forward_average, opt.scale))

    