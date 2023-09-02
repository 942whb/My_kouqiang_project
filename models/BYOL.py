import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_base_network import ResNet18
import os
class BYOL_res(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
#todo： 怎么加载预训练模型       
        self.encoder = self.encoder1
        self.logdir = torch.nn.Linear(1024, 1)
    def forward(self, x):
#todo:加载预训练模型，以及后面跟上一个简单的分类器就行，记得要最后加上一个 sigmod(),
        feature = self.encoder(x)
        out = self.logdir(feature)
        out = nn.Sigmoid(out)
        return out


    def encoder1(self,feature):
        encoder = ResNet18('resnet18')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #这里加载预训练模型即可
        load_params = torch.load(os.path.join('/home/whb/PyTorch-BYOL-master/kouqiang/checkpoints/model.pth'),
                         map_location=torch.device(torch.device(device)))
        if 'online_network_state_dict' in load_params:
            encoder.load_state_dict(load_params['online_network_state_dict'])
        print("Parameters successfully loaded.")

# remove the projection head
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
        encoder = encoder.to(device)
        encoder.eval()
        return encoder(feature)