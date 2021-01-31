import torch
import torch.nn as nn
import torch.nn.functional as F

#from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2_dw import MobileNetV2
from models.dynamic_mobilenetv2 import DyMobileNetV2 
from models.ensemble import DynamicEnsemble
from models.resnet18 import ResNet18


def build_models(model_name, num_classes, input_size, model_size,
                 pretrained=False, model_path=None):

    model_size = model_size.upper()
    assert model_size in ['S', 'M']
    width_mult = {'S' : 0.5, 'M' : 1}
    
    if model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes, input_size=input_size, width_mult=width_mult[model_size])
    elif model_name == 'resnet18':
        model = ResNet18()
    return model
