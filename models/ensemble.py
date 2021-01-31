import sys
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from models.mobilenetv2 import MobileNetV2
from models.dynamic_mobilenetv2 import DyMobileNetV2
from utils.general import select_device

model_paths = [
      'runs/train/mobilenet/weights/mobilenetv2epoch_100.pth',
      'runs/train/mobilenet2/weights/mobilenetv2epoch_100.pth',
      'runs/train/mobilenet3/weights/mobilenetv2epoch_100.pth',
    ]

class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, dilation=1, groups=1, bias=False, 
                 num_models=None, module_types=None):
        super(KernelAttention, self).__init__()
        if num_models is None or module_types is None:
            raise ValueError('num_models and module_types must named')
            
        self.conv = nn.Conv2d(in_channels, out_channels)
    def forward(self, x):
        pass

class DynamicEnsemble(nn.Module):
    def __init__(self, temperature, num_classes=10, pretrained=False):
        super(DynamicEnsemble, self).__init__()
        K = len(model_paths)
        self.pretrained = pretrained
        self.models = self.build(K, temperature, num_classes, model_paths)
    
    def build(self, K, temperature, num_classes, model_paths):
        model = DyMobileNetV2(K, temperature, num_classes=num_classes)
        m_dict = model.state_dict()
        if self.pretrained:
            pre_trained_m_dicts = [torch.load(path) for path in model_paths]
            for key in m_dict:
                if key.find('attention') != -1:
                    continue
                elif m_dict[key].dim() == 5 and pre_trained_m_dicts[0][key].dim() == 4:
                    for i in range(len(pre_trained_m_dicts)):
                        m_dict[key][i] = pre_trained_m_dicts[i][key]
                else:
                    if m_dict[key].dim() == 0:
                        continue
                    temp_weights = [pre_trained_m_dicts[i][key].unsqueeze(0) for i in range(len(pre_trained_m_dicts))]
                    temp_weights = torch.cat(temp_weights, dim=0).mean(dim=0)
                    m_dict[key] = temp_weights
        model.load_state_dict(m_dict)
        print("Sucessfully overwritting !!")
        return model

    def forward(self, x):
        return self.models(x)

    
class BaseEnsemble(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(BaseEnsemble, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_models = len(model_paths)
        self.models = self.build_models(model_paths)
      
    def build_models(self, model_paths):
        m = []
        for path in model_paths:
            model = MobileNetV2(self.num_classes)
            if self.pretrained:
                model.load_state_dict(torch.load(path))
            m.append(model)
        return nn.ModuleList(m)

    
    def forward(self, x, inference=True):
        y = []
        for m in self.models:
            y.append(F.softmax(m(x),dim=1).unsqueeze(1))
        out = torch.cat(y, dim=1).sum(dim=1)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = select_device(opt.device)

    model = DynamicEnsemble(pretrained=True).to(device)

            
