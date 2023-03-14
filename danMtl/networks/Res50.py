from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from danMtlTrt.networks.resnet import ResNet
import numpy as np
import torch
from torchvision import transforms

class res50(nn.Module) :
    def __init__(self,num_class=2):
        super(res50, self).__init__()
        N_IDENTITY = 8631
        include_top = True 
        resnet = ResNet.resnet50(pretrained_checkpoint_path="./danMtlTrt/models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(2048, num_class)
        
        self.sig = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.squeeze()
        x = x.unsqueeze(0)
        out = self.sig(self.fc1(x))

        return out