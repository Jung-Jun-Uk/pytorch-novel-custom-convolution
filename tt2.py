import torch
import torch.nn as nn
from models.mobilenetv2_proposed import MobileNetV2

model = MobileNetV2(num_classes=1000, input_size=224)

x = torch.randn(1,3,224,224)
out = model(x)
print(out.size())



