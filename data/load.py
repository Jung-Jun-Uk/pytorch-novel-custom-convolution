from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet

def load_datasets(dataset_name, batch_size, cuda, workers, rank):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(batch_size, cuda, workers, rank)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(batch_size, cuda, workers, rank)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(batch_size, cuda, workers, rank)
    return dataset
