import os
import sys
import datetime
import time

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.load import load_datasets
from models.build import build_models
from utils.general import select_device, increment_path, Logger, AverageMeter, save_model, \
    print_argument_options, init_torch_seeds, mkdir_if_missing


def main(opt, device):

    if not opt.nlog and not opt.test:
        sys.stdout = Logger(Path(opt.save_dir) / 'log_.txt')
    
    if opt.global_rank in [-1, 0]:
        print_argument_options(opt)
    
    #Configure
    cuda = device.type != 'cpu'
    init_torch_seeds()

    dataset = load_datasets(opt.data, opt.batch_size, cuda, opt.workers, opt.global_rank)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    opt.num_classes = dataset.num_classes

    if opt.global_rank in [-1, 0]:
        print("Creat dataset: {}".format(opt.data))

    model = build_models(opt.model, opt.num_classes, opt.input_size, opt.model_size,
                         opt.pretrained).to(device)
    
    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    if cuda and opt.global_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    if opt.global_rank in [-1, 0]:
        print(model)
        print("Creat model: {}".format(opt.model))


    if opt.test:
        model.load_state_dict(torch.load('runs/train/cifar10/mobilenetv2(S)/dynamic_base/weights/best_epoch_1.pth'))
        acc, err = __testing(opt, model, testloader, 0, device)
        print("==> Train Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
        return 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, weight_decay=1e-04, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1)
    opt.scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_acc = 0

    if opt.global_rank in [-1, 0]:
        start_time = time.time()    
    for epoch in range(opt.max_epoch):
        if opt.global_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        
        if opt.global_rank in [-1, 0]:
            print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))
        __training(opt, model, criterion, optimizer, trainloader, epoch, device, opt.global_rank)
        scheduler.step()

        if opt.eval_freq > 0 and (epoch+1) % opt.eval_freq == 0 or (epoch+1) == opt.max_epoch:
            #acc, err = __testing(opt, model, trainloader, epoch, device)
            #if opt.global_rank in [-1, 0]:
            #    print("==> Train Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            acc, err = __testing(opt, model, testloader, epoch, device)
            if opt.global_rank in [-1, 0]:
                print("==> Test Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
                
                if acc > best_acc:
                    save_model(model, 0, name='best', save_dir=opt.save_dir)
                    best_acc = max(best_acc, acc)

    if opt.global_rank in [-1, 0]:
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def __training(opt, model, criterion, optimizer, trainloader, epoch, device, rank):
    model.train()
    losses = AverageMeter()
    
    start_time = time.time() 
    for i, (data, labels) in enumerate(trainloader):
    
        data, labels = data.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, labels)
            
        opt.scaler.scale(loss).backward()
        opt.scaler.step(optimizer)
        opt.scaler.update()
        optimizer.zero_grad()

        losses.update(loss.item(), labels.size(0))
                 
        if (i+1) % opt.print_freq == 0 and rank in [-1, 0]:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, elapsed))
            

def __testing(opt, model, testloader, epoch, device):
    model.eval()
    correct, total = 0, 0
                
    with torch.no_grad():
        for i, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.to(device)
            """ if i == 0 or i == 10:
                o = model(data)
                pav = model.attention_plotting() # plot attention values
                #print(pav[0].cpu().data)
                pav = torch.transpose(pav[0], 0, 1)
                ax = sns.heatmap(pav.cpu().data, cmap='viridis', linewidth=0.3, vmin=0, vmax=1)

                dirname = os.path.join(opt.save_dir, 'plots')
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                plt.savefig(os.path.join(dirname, 'image_' + str(i) + '_' + str(epoch+1) +'.png'))
                plt.close() """

            outputs = model(data)    
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr'               , type=int, default=0.1)
    parser.add_argument('--workers'          , type=int, default=4)
    parser.add_argument('--batch_size'       , type=int, default=512)
    parser.add_argument('--max_epoch'        , type=int, default=200)
    parser.add_argument('--gamma'            , type=int, default=0.1)
    parser.add_argument('--input_size'       , type=int, default=32)
    parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

    parser.add_argument('--eval_freq'        , default=1)
    parser.add_argument('--print_freq'       , default=50)
    parser.add_argument('--num_classes'      , default=10)  
    parser.add_argument('--data'             , choices=['cifar10', 'cifar100'])
    parser.add_argument('--model'            , choices=['mobilenetv2', 'resnet18'])
    parser.add_argument('--model_size'       , type=str, default='s', help= 'model_size S(s), M(m)')
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--test',action='store_true', help='just test')

    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    return parser.parse_args()

if __name__ == "__main__":
    # python -m torch.distributed.launch --nproc_per_node train_temp.py
    opt = parser()
    opt.save_dir = increment_path(Path(opt.project) / opt.data / (opt.model + '(' + opt.model_size.upper() +')') / opt.name, exist_ok=opt.exist_ok)  # increment run
    
    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    #DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size, rank=opt.global_rank)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        #dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:631', world_size=opt.world_size, rank=opt.local_rank)  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    main(opt, device)

    
