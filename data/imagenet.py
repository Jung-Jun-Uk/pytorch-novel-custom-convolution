import os
import sys
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import random
import numpy as np
from PIL import Image
import pickle
#from RandAugment import RandAugment

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from data.utils import create_train_imagenet_dataset, create_val_imagenet_dataset


def name2category_map(path):
    name2category = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name2category[line[:9].strip()] = line[10:].strip()
    return name2category

def name2label_map(path):
    name2label = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            name2label[line[:9].strip()] = i
    return name2label


class ImageNetDataset(data.Dataset):
    def __init__(self, ilsvrc_path='/data/ssd1/jju/ImageNet/ILSVRC', 
                 loc_synset_mapping_path='data/store/LOC_synset_mapping.txt',
                 save_dir ='data/store',
                 train=True,
                 sample_class=100):
        crop_size = 224
        val_size = 256
        self.name2label = name2label_map(loc_synset_mapping_path)
        self.train = train

        if self.train:
            file_name = os.path.join(save_dir, 'imagenet_train' + '.pickle')
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as f:
                    information = pickle.load(f)
            else:
                information = create_train_imagenet_dataset(ilsvrc_path=ilsvrc_path)
                with open(file_name, 'wb') as f:
                    pickle.dump(information, f)
        else:
            file_name = os.path.join(save_dir, 'imagenet_test' + '.pickle')
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as f:
                    information = pickle.load(f)
            else:
                information = create_val_imagenet_dataset(ilsvrc_path=ilsvrc_path)
                with open(os.path.join(save_dir, 'imagenet_test' + '.pickle'), 'wb') as f:
                    pickle.dump(information, f)

        self.information = information

        if sample_class is not None:
            name = open('data/store/sample_class100.txt', 'r').read().split('\n')
            #name = list(self.name2label.keys())
            name = name[:sample_class]
            self.name2label = {n : i for i, n in enumerate(name)}
            self.information = self.sampling()
            
        """ with open('data/store/sample_class100.txt', 'w+') as f:
            f.write('\n'.join(name)) """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose([
            #transforms.Resize((size,size)),
            transforms.RandomResizedCrop((crop_size,crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        
        #self.train_transform.transforms.insert(0, RandAugment(2, 9))

        self.val_transform = transforms.Compose([
            transforms.Resize((val_size,val_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop((crop_size,crop_size)),
            transforms.ToTensor(),
            normalize,
            ])

    def sampling(self):
        print("Sampling processing")
        sample_set = set(self.name2label.keys())
        sampled_information = []
        for info in self.information:
            if info['name'] in sample_set:
                sampled_information.append(info)
        print("sampling done!, {} classes".format(len(sample_set)))
        return sampled_information

    def __getitem__(self, index):
        info = self.information[index]
        img = Image.open(info['path']).convert('RGB')
        if self.train:
            data = self.train_transform(img)
        else:
            data = self.val_transform(img)
        label = self.name2label[info['name']]
        return data, label

    def __len__(self):
        return len(self.information)


class ImageNet(object):
    def __init__(self, batch_size, cuda, workers, rank):
        #root= '/data/'
        if rank in [-1, 0]:
            print("ImageNet processing .. ")
        train_dataset = ImageNetDataset(train=True, sample_class=None)
        val_dataset = ImageNetDataset(train=False, sample_class=None)
        
        pin_memory = True if cuda else False
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if rank != -1 else None

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=pin_memory, sampler=train_sampler)

        valloader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
            )
                                    
        self.num_classes = 1000
        self.trainloader = trainloader
        self.testloader = valloader

        if rank in [-1, 0]:
            print("len trainloader", len(self.trainloader))
            print("len testloader", len(self.testloader))


if __name__ == "__main__":
    ImageNetDataset(train=True)
    #ImageNetDataset(train=False)
