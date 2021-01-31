import os
import sys
import datetime
import time

import argparse
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.load import load_datasets
from models.build import build_models
from loss.smooth_ce import SmoothCrossEntropyLoss

from utils.general import select_device, increment_path, Logger, AverageMeter, save_model, \
    print_argument_options, init_torch_seeds


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'model_best.pth.tar'))

def main(opt, device):
    best_acc1 = 0
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

    model = build_models(opt.model, opt.num_classes, opt.input_size, opt.model_size).to(device)
    
    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    if cuda and opt.global_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    
    if opt.global_rank in [-1, 0]:
        print(model)
        print("Creat model: {}".format(opt.model))

    criterion = nn.CrossEntropyLoss()
    #criterion = SmoothCrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, weight_decay=5e-04, momentum=0.9)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if opt.global_rank in [-1, 0]:
                print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            if opt.global_rank in [-1, 0]:
                print("=> no checkpoint found at '{}'".format(opt.resume))

    opt.scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    if opt.global_rank in [-1, 0]:
        start_time = time.time()    
    for epoch in range(opt.start_epoch, opt.max_epoch):
        if opt.global_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        if opt.global_rank in [-1, 0]:
            print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))
        __training(opt, model, criterion, optimizer, trainloader, epoch, device, opt.global_rank)
    
        if opt.eval_freq > 0 and (epoch+1) % opt.eval_freq == 0 or (epoch+1) == opt.max_epoch:
            #if cuda and opt.global_rank != -1:
            #    model.module.inference_mode()
            #else:
            #    model.inference_mode()
            acc1 = __testing(opt, model, testloader, epoch, device, opt.global_rank)

            #if cuda and opt.global_rank != -1:
            #    model.module.training_mode()
            #else:
            #    model.training_mode()
            acc1 = __testing(opt, model, testloader, epoch, device, opt.global_rank)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)    
            if opt.global_rank in [-1, 0]:
                save_checkpoint({
                'epoch': epoch + 1,
                'arch': opt.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, save_dir=opt.save_dir)
                

    if opt.global_rank in [-1, 0]:
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def __training(opt, model, criterion, optimizer, trainloader, epoch, device, rank):
    model.train()
    losses = AverageMeter()
    
    start_time = time.time() 
    trainloader_len = len(trainloader)
    for i, (data, labels) in enumerate(trainloader):
        adjust_learning_rate(opt, optimizer, epoch, i, trainloader_len)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, labels)
            
        opt.scaler.scale(loss).backward()
        opt.scaler.step(optimizer)
        opt.scaler.update()
        

        losses.update(loss.item(), labels.size(0))
                 
        if (i+1) % opt.print_freq == 0 and rank in [-1, 0]:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, elapsed))
            

def __testing(opt, model, testloader, epoch, device, rank):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()            
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], labels.size(0))
            top5.update(acc5[0], labels.size(0))
    
    if rank in [-1, 0]:
        print("Acc@1 {:.3f} Acc@5 {:.3f}".format(top1.avg, top5.avg))
    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


from math import cos, pi
def adjust_learning_rate(opt, optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if opt.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = opt.max_epoch * num_iter

    if opt.lr_decay == 'step':
        lr = opt.lr * (opt.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif opt.lr_decay == 'cos':
        lr = opt.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif opt.lr_decay == 'linear':
        lr = opt.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif opt.lr_decay == 'schedule':
        count = sum([1 for s in opt.schedule if s <= epoch])
        lr = opt.lr * pow(opt.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_decay))

    if epoch < warmup_epoch:
        lr = opt.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr'               , type=float, default=0.05)
    parser.add_argument('--workers'          , type=int, default=4)
    parser.add_argument('--batch_size'       , type=int, default=256)
    parser.add_argument('--max_epoch'        , type=int, default=150)
    parser.add_argument('--start_epoch'      , type=int, default=0)
    parser.add_argument('--gamma'            , type=int, default=0.1)
    parser.add_argument('--lr_decay'         , type=str, default='cos', help='mode for learning rate decay')
    parser.add_argument('--input_size'       , type=int, default=224)
    parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

    parser.add_argument('--eval_freq'        , default=1)
    parser.add_argument('--print_freq'       , default=50)
    parser.add_argument('--num_classes'      , default=1000)  
    parser.add_argument('--data'             , type=str, default='imagenet')
    parser.add_argument('--model'            , choices=['mobilenetv2'])
    parser.add_argument('--model_size'       , type=str, default='s', help= 'model_size S(s), M(m), L(l)')
    parser.add_argument('--test',action='store_true', help='just test')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    return parser.parse_args()

if __name__ == "__main__":
    # python -m torch.distributed.launch --nproc_per_node 4 train_temp.py
    opt = parser()
    opt.save_dir = increment_path(Path(opt.project) / opt.data / (opt.model + '(' + opt.model_size.upper() +')') / opt.name, exist_ok=opt.exist_ok)  # increment run
    if opt.resume:
        opt.resume = os.path.join(opt.save_dir, 'checkpoint.pth.tar')
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