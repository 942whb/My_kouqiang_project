

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:23 2020

@author: zhang
"""

import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import math
from sobel import Gedge_map
from sobel import edge_conv2d128, edge_conv2d64
from sobel import edge_conv2d256
from vit_seg_configs import get_b16_config
from vit_class_axisgate_atten import fuseTransformer
from fre_atten_pro import MultiSpectralAttentionLayer
nonlinearity = partial(F.relu, inplace=True)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        out = residual + out
        out = self.relu(out)
        return out
    

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    


class local_attention(nn.Module):
    def __init__(self, channel):
        super(local_attention, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel // 2, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel // 2, 1, kernel_size=1, dilation=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.u1 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        self.u2 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.psi = nn.Sequential(
            nn.Conv2d(channel // 2, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, H, W = x.size()
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(self.dilate1(x)))
        dilate3_out = nonlinearity(self.dilate3(self.dilate2(self.dilate1(x))))

        fea1 = dilate1_out
        fea2 = dilate2_out
        fea3 = dilate3_out

        fea = fea1+fea2+fea3

        edgemap = self.relu(Gedge_map(self.psi(fea))+self.psi(fea))

        x = x*edgemap
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)+x

class shallow_fea_fusion(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(shallow_fea_fusion,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)


        self.upsam = up_conv(ch_in=128, ch_out=64)
        self.shallow_conv = conv_block(ch_in=128, ch_out=64)
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g = self.upsam(g)
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1+x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        fea1 = x*psi
        fea2 = g*psi
        fea = torch.cat((fea1,fea2),dim=1)
        fea = self.shallow_conv(fea)
        fea = self.conv1x1(fea)
        return fea

class muti_fusion(nn.Module):
    def __init__(self):
        super(muti_fusion,self).__init__()
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Maxpool8 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.conv = conv_block(ch_in=480, ch_out=240)
    def forward(self, fea1, fea2, fea3, fea4):

        fea = torch.cat((self.Maxpool8(fea1), self.Maxpool4(fea2), self.Maxpool2(fea3), fea4),dim=1)
        fea = self.conv(fea)
        return fea

class U_Net(nn.Module):
    def __init__(self, in_c, n_classes):
        super(U_Net, self).__init__()
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = conv_block(ch_in=3, ch_out=32)#512*512
        self.Conv2 = conv_block(ch_in=32, ch_out=32)#256*256
        self.Conv3 = conv_block(ch_in=64, ch_out=64)#128*128
        self.Conv4 = conv_block(ch_in=128, ch_out=128)#64*64

        self.edgat2 = MultiSpectralAttentionLayer(32,128,128, freq_sel_method = 'top16')
        self.edgat3 = MultiSpectralAttentionLayer(64,64,64, freq_sel_method = 'top16')
        self.edgat4 = MultiSpectralAttentionLayer(128,32,32, freq_sel_method = 'top16')
        
        self.fusion = muti_fusion()

        self.Trans = fuseTransformer(get_b16_config(), img_size=64, inchannel=256, outchannel=256, ychannel=128)

        self.pool_tofc = nn.AdaptiveMaxPool2d((1,1))
        layers = [
            nn.Linear(256, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1)
        x2_1 = self.Conv2(x2)
        x2_2 = self.edgat2(x2)
        x2 = torch.cat((x2_1,x2_2),dim=1)

        x3 = self.Maxpool(x2)
        x3_1 = self.Conv3(x3)
        x3_2 = self.edgat3(x3)
        x3 = torch.cat((x3_1,x3_2),dim=1)

        x4 = self.Maxpool(x3)
        x4_1 = self.Conv4(x4)
        x4_2 = self.edgat4(x4)
        x4 = torch.cat((x4_1,x4_2),dim=1)
        #print(x.shape,x1.shape,x2.shape,x3.shape,x4.shape)#[8,3,512,512] [8,3,512,512] [8,4,256,256] [8,128,128,128] [8,256,64,64]
        x7 = self.Trans(x4)
        
        d1 = self.pool_tofc(x7)
        d1 = d1.squeeze(dim=3).squeeze(dim=2)
        out = self.layers(d1)

        return out