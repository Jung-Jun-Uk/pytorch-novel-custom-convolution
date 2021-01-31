import os
import sys
import datetime
import time

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data.load import load_datasets
from models.build import build_models
from utils.general import select_device, increment_path, Logger, AverageMeter, save_model, \
    print_argument_options, init_torch_seeds


def main(opt, device):

    if not opt.nlog and not opt.test:
        sys.stdout = Logger(Path(opt.save_dir) / 'log_.txt')
    print_argument_options(opt)
    
    #Configure
    cuda = device.type != 'cpu'
    init_torch_seeds()

    dataset = load_datasets(opt.data, opt.batch_size, cuda, opt.workers)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    opt.num_classes = dataset.num_classes
    print("Creat dataset: {}".format(opt.data))

    model = build_models(opt.model, opt.num_classes).to(device)
    print(model)
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("Creat model: {}".format(opt.model))

    if opt.test:
        acc, err = __testing(opt, model, testloader, 0, device)
        print("==> Train Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
        return 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, weight_decay=5e-04, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
    
    if opt.amp:
        opt.scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_time = time.time()    
    for epoch in range(opt.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))
        __training(opt, model, criterion, optimizer, trainloader, epoch, device)
        scheduler.step()

        if opt.eval_freq > 0 and (epoch+1) % opt.eval_freq == 0 or (epoch+1) == opt.max_epoch:
            acc, err = __testing(opt, model, trainloader, epoch, device)
            print("==> Train Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            acc, err = __testing(opt, model, testloader, epoch, device)
            print("==> Test Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            save_model(model, epoch, name=opt.model, save_dir=opt.save_dir)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def __training(opt, model, criterion, optimizer, trainloader, epoch, device):
    model.train()
    losses = AverageMeter()
    
    start_time = time.time() 
    for i, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        
        if opt.amp: #amp autocast
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
            opt.scaler.scale(loss).backward()
            opt.scaler.step(optimizer)
            opt.scaler.update()
        
        else:
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), labels.size(0))
                 
        if (i+1) % opt.print_freq == 0:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, elapsed))
            

def __testing(opt, model, testloader, epoch, device):
    model.eval()
    correct, total = 0, 0
                
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err
    

def parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr'               , default=0.1)
    parser.add_argument('--workers'          , default=4)
    parser.add_argument('--batch_size'       , default=256)
    parser.add_argument('--max_epoch'        , default=100)
    parser.add_argument('--stepsize'         , default=30)
    parser.add_argument('--gamma'            , default=0.1)
    parser.add_argument('--eval_freq'        , default=10)
    parser.add_argument('--print_freq'       , default=50)
    parser.add_argument('--num_classes'      , default=10)  
    parser.add_argument('--data'             , choices=['cifar10', 'cifar100'])
    parser.add_argument('--model'            , choices=['mobilenetv2', 'mobilenetv2_dy_depth', 'mobilenetv2_dy_depth_group', 'dy_mobilenetv2', 'dy_ensemble', 'mobilenetv2_dyd'])
    parser.add_argument('--test',action='store_true', help='just test')

    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--amp', action='store_true', help='using amp.autocast, it is more faster')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parser()
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.save_dir = increment_path(Path(opt.project) / opt.data / opt.model / opt.name, exist_ok=opt.exist_ok)  # increment run
    
    main(opt, device)

    
