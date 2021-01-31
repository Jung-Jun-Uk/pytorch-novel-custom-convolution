import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import random


class CIFAR10(object):
    def __init__(self, batch_size, cuda, num_workers, rank):
        root= '/data/'
        
        pin_memory = True if cuda else False
        
        img_size = 32
        padding = 4
        transform_train = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomCrop(img_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                               transform=transform_test)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset) if rank != -1 else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset) if rank != -1 else None

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),
                                                  num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=pin_memory, sampler=test_sampler)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(self.classes)
        self.trainloader = trainloader
        self.testloader = testloader
        
        if rank in [-1, 0]:
            print("len trainloader", len(self.trainloader))
            print("len testloader", len(self.testloader))


class CIFAR100(object):
    def __init__(self, batch_size, cuda, num_workers):
        root= '/data/'
        
        pin_memory = True if cuda else False
        
        img_size = 32
        padding = 4
        transform_train = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomCrop(img_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                                transform=transform_train)
    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=pin_memory)
                                    
        #self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = 100
        self.trainloader = trainloader
        self.testloader = testloader
        print("len trainloader", len(self.trainloader))
        print("len testloader", len(self.testloader))        