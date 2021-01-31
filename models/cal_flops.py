import argparse
import math
import time
import sys
import torch
import os

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from models.build import build_models
from utils.general import select_device
from ptflops import get_model_complexity_info
from torchstat import stat
from torchsummaryX import summary

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-f', '--features', type=int, default=32)
    parser.add_argument('-s', '--state-size', type=int, default=128)
    parser.add_argument('-r', '--runs', type=int, default=100)
    parser.add_argument('--model', choices=['mobilenetv2'])
    parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
    #parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parser()
    #device = select_device(opt.device, batch_size=opt.batch_size)
    #X = torch.randn(opt.batch_size, 3, opt.features, opt.features).to(device)
    
    model = build_models(opt.model, num_classes=1000, input_size=224, model_size='S')
    #stat(model, (3,224,224))
    summary(model, torch.zeros(1,3,224,224))
    #with torch.cuda.device(0):
    #    net = build_models(opt.model, 10)
        #net.optim_forward = True
        #macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
        #                                   print_per_layer_stat=True, verbose=True)
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
