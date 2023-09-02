#构建一个我自己的模型，要达到通过 resnet 学习局部特征，通过 attention 学习全局特征，通过 gcn学习结构特征
#to do:去找三个东西，然后拼在一起，学习得到三个feature之后拼接在一起，之后再通过fc网络达到分类和分割的效果
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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from .vit import ViT
import math
import cv2
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np
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
    
class ReDense_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ReDense_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 2*ch_in, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2*ch_in)
        self.conv2 = nn.Conv2d(3*ch_in, 4*ch_in, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4*ch_in)
        self.conv3 = nn.Conv2d(7*ch_in, ch_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(ch_out)
        self.bnre = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        res = self.conv1x1(x)
        res = self.bnre(res)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out1 = self.drop(out1)
        out = torch.cat((x,out1),dim=1)
        out2 = self.relu(self.bn2(self.conv2(out)))
        out2 = self.drop(out2)
        out = torch.cat((out1,out2,x),dim=1)
        out = self.relu(self.bn3(self.conv3(out)))
        out = res + out
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
    
    
    
class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout,inputsize):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim#要设置的两个参数，一个是输出维度等
        self.n_heads = n_heads#头的个数
        self.inputsize  = inputsize
        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0#也就是这个hid_dim 是相当于最后四个头拼接起来的一个向量
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(inputsize, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(inputsize, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(inputsize, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.trans = nn.Conv2d(inputsize,16,kernel_size=3,stride=1,padding=1)
        self.shape = nn.Conv2d(3,1,kernel_size=1,stride=1,padding=0)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, img, mask=None):
        #我这里的输入是x 为 [batch,512,512]的一个这个样子的图像，首先应该要做一个类似于embedding的操作？
        # x先通过一个卷积层？ 把它变成一个 [batch,16,256]的结构？
        #再拆成[batch,16,4,64]
        #然后就是通过后面的mutihead attention得到一个 
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        img = self.shape(img)
        img = img.squeeze()
        #print(img.shape)
        bsz = img.shape[0]
        Q = self.w_q(img)
        K = self.w_k(img)
        V = self.w_v(img)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(device='cuda')

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        #[8,inputsize,256] 这里得做点东西，把它先变成[8,inputsize*256]?
        #我再这一步的结果也是 [batch,16,feature],我得看看之前那个cnn得到的feature是什么维度，之前的是256,所以我这个也设置成256
        #所以我这里应该是要把[batch,16,256] 变成[batch,256],用mean就行,这个参数太少了，类似于什么情况呢，应该是把
        x = x.mean(dim = 1)
        x = x.squeeze()
        return x

class quanzhong(nn.Module):
    def _init_(self,in_c,out_c):
        super(quanzhong,self)._init_()
        self.linear = nn.Linear(in_c,out_c,bias=False)
        
    def forward(self,x):
        return self.linear(x)
    
    def get_num(self):
        return self.linear.weight
#class GraphCNN():
#todo:找一个图卷积神经网络放进来，用来获得纹理特征

class My_Model_end(nn.Module):
    def __init__(self, in_c, n_classes):
        super(My_Model_end,self).__init__()
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Maxpool_img = nn.AvgPool2d(kernel_size = 2,stride = 2)
        self.Conv1 = conv_block(ch_in=3, ch_out=32)#512*512
        self.Conv2 = conv_block(ch_in=32, ch_out=32)#256*256
        self.Conv3 = conv_block(ch_in=64, ch_out=64)#128*128
        self.Conv4 = conv_block(ch_in=128, ch_out=128)#64*64
        self.Conv5 = conv_block(ch_in=256, ch_out=256)#???
        self.Seg_Conv1 = conv_block(ch_in=1, ch_out=9)#512*512
        self.Seg_Conv2 = conv_block(ch_in=9, ch_out=16)#256*256
        self.Seg_Conv3 = conv_block(ch_in=16, ch_out=32)#128*128
        self.Seg_Conv4 = conv_block(ch_in=128, ch_out=128)
        self.Conv6 =conv_block(ch_in=512, ch_out=256)
        self.Conv7 =conv_block(ch_in=256, ch_out=128)
        self.Conv8 =conv_block(ch_in=128, ch_out=64)
        self.Conv9 =conv_block(ch_in=64, ch_out=32)
        self.Conv10 =conv_block(ch_in=32, ch_out=16)
        
        self.edgat2 = MultiSpectralAttentionLayer(32,128,128, freq_sel_method = 'top16')
        self.edgat3 = MultiSpectralAttentionLayer(64,128,128, freq_sel_method = 'top16')
        self.edgat4 = MultiSpectralAttentionLayer(128,128,128, freq_sel_method = 'top16')
        self.edgat5 = MultiSpectralAttentionLayer(256,128,128, freq_sel_method = 'top16')
        #self.edgat6 = MultiSpectralAttentionLayer(128,128,128, freq_sel_method = 'top16')
        self.fusion = muti_fusion()

        self.Trans = fuseTransformer(get_b16_config(), img_size=64, inchannel=256, outchannel=256, ychannel=128)
        self.Trans_D = fuseTransformer(get_b16_config(), img_size=256, inchannel=16, outchannel=16, ychannel=128)

        self.pool_tofc = nn.AdaptiveMaxPool2d((1,1))
        layers = [
            nn.Linear(256, 1)
            #nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
        self.liner = nn.Linear(16,1,bias=False)
        #self.attention_512 = MultiheadAttention(hid_dim=256,n_heads=4,dropout=0.15,inputsize=512)
        #self.attention_256 = MultiheadAttention(hid_dim=256,n_heads=4,dropout=0.15,inputsize=256)
        #self.attention_128 = MultiheadAttention(hid_dim=256,n_heads=4,dropout=0.15,inputsize=128)
        #self.attention_64 = MultiheadAttention(hid_dim=256,n_heads=4,dropout=0.15,inputsize=64)
        #self.attention_512 = ViT(image_size = 512,patch_size = 32,num_classes = 1,dim = 256,depth = 6,heads = 4,mlp_dim = 1024,dropout = 0.1,emb_dropout = 0.1)
        #self.attention_256 = ViT(image_size = 256,patch_size = 32,num_classes = 1,dim = 256,depth = 6,heads = 4,mlp_dim = 1024,dropout = 0.1,emb_dropout = 0.1)
        #self.attention_128 = ViT(image_size = 128,patch_size = 32,num_classes = 1,dim = 256,depth = 6,heads = 4,mlp_dim = 1024,dropout = 0.1,emb_dropout = 0.1)
        #self.attention_64 = ViT(image_size = 64,patch_size = 16,num_classes = 1,dim = 256,depth = 6,heads = 4,mlp_dim = 1024,dropout = 0.1,emb_dropout = 0.1)
        #self.fc = nn.Sequential(
        #    nn.Linear(4*256, 256),
        #    nn.LayerNorm(256)
        #) 
        #self.fc1 = nn.Sequential(
        #    nn.Linear(256, 1),
        #    nn.Sigmoid()
        #)
        
        self.feature_ln = nn.LayerNorm(256)
        #self.attention_512_ln = nn.LayerNorm(256)
        #self.attention_256_ln = nn.LayerNorm(256)
        #self.attention_128_ln = nn.LayerNorm(256)
        #self.attention_64_ln = nn.LayerNorm(256)
        
        self.Dense_1 = conv_block(ch_in=3, ch_out=32)# (3,512,512)->(6,512,512)
        self.Dense_2 = conv_block(ch_in=32, ch_out=64)# (6,512,512)->(9,512,512)
        self.Dense_3 = conv_block(ch_in=64, ch_out=128)
        self.Dense_4 = conv_block(ch_in=128, ch_out=12)#(9,512,512)->(12,512,512)   (256,32,32)->(256,1,1)
        self.pool_tofc_D = nn.AdaptiveAvgPool2d((1,1))
        self.feature_D = nn.LayerNorm(256)
        self.sig = nn.Sigmoid()
#换一个思路
    def forward(self, x,seg):# seg是 [batch,1,512,512] -> 通过一个CNN单独引入到
        #img_512 = x
        #img_256 = self.Maxpool_img(img_512)
        #img_128 = self.Maxpool_img(img_256)
        #img_64 = self.Maxpool_img(img_128)
        #att_512 = self.attention_512(img_512)
        #att_512 = self.attention_512_ln(att_512)
        #att_256 = self.attention_256(img_256)
        #att_256 = self.attention_256_ln(att_256)
        #att_128 = self.attention_128(img_128)
        #att_128 = self.attention_128_ln(att_128)
        #att_64 = self.attention_64(img_64)
        #att_64 = self.attention_64_ln(att_64)
        #att = torch.cat([att_512,att_256,att_128,att_64],dim = 1,out=None)
        #att_feature = self.fc(att)
        x_ = []
        #x1 = self.Conv1(x) 
        #x_.append(x1)
        #x2 = self.Maxpool(x1)
        #x2_1 = self.Conv2(x2)
        #x2_2 = self.edgat2(x2)
        #x2 = torch.cat((x2_1,x2_2),dim=1)
        #x_.append(x2)
        #x3 = self.Maxpool(x2)
        #x3_1 = self.Conv3(x3)
        #x3_2 = self.edgat3(x3)
        #x3 = torch.cat((x3_1,x3_2),dim=1)
        #x_.append(x3)
        #x4 = self.Maxpool(x3)
        #x4_1 = self.Conv4(x4)
        #x4_2 = self.edgat4(x4)
        #x4 = torch.cat((x4_1,x4_2),dim=1)
        #x_.append(x4)
        # x1,x2,x3,x4都返回print 出来一下，resize,channel数有点多，到时候统一分成八个
        #print(x.shape,x1.shape,x2.shape,x3.shape,x4.shape)#[8,3,512,512] [8,32,512,512] [8,64,256,256] [8,128,128,128] [8,256,64,64]
        #x7 = self.Trans(x4)
        #d1 = self.pool_tofc(x7)
        #print(x7.shape,d1.shape)#torch.Size([batch, 256, 16, 16]) torch.Size([batch, 256, 1, 1])
        #总共改了三个地方，加载数据的增强，pooling的方式，以及最终的参数，加了最后一个之后，模型开始往1坍缩
        #seg = ....  构建一个seg的东西
        # python train_model_new.py --csv_train data/data2/train.csv --epoch_num 500 --model_name MY_new  --save_path classmodel_new_exp_pad --im_size 512 --batch_size 1 --grad_acc_steps 1 --device cuda:1 
        seg = seg.squeeze(1)
        seg = torch.round(seg)
        pre = seg
        #print(seg.sum())
        seg = seg.squeeze(0)
        seg = seg.cpu().detach().numpy()
        seg = seg.astype(np.uint8)
        a,b = 512,512
        c,d = 0,0 
        for i in range(512):
            for j in range(512):
                #print(seg[i,j])
                if seg[i,j] > 0.6:
                    a = min(a, i)
                    b = min(b, j)
                    c = max(c, i)
                    d = max(d, j)
        dif = max(c-a,d-b)
        #print(a,b,c,d,dif)
        seg_input = (x[:,:,max(0,a-30):min(a+dif+30,511),max(0,b-30):min(b+dif+30,511)])
        #todo:增加一个Loss
        #print(seg_input.shape)
        #inputs_c1_np = x.squeeze().permute(1, 2, 0).cpu().numpy() 
        #img0_np = seg_input.squeeze().permute(1, 2, 0).cpu().numpy()
        #cv2.imwrite('inputs_c1.png', inputs_c1_np*255.0)
        #cv2.imwrite('img0.png', img0_np*255.0)
        #seg_input = torch.tensor(seg_input)
        x1 = self.Conv1(seg_input)
        x_.append(x1)
        #x2 = self.Maxpool(x1)
        x2_1 = self.Conv2(x1)
        x2_2 = self.edgat2(x1)
        x2 = torch.cat((x2_1,x2_2),dim=1)
        #x_.append(x2)
        #x3 = self.Maxpool(x2)
        x3_1 = self.Conv3(x2)
        x3_2 = self.edgat3(x2)
        x3 = torch.cat((x3_1,x3_2),dim=1)
        #x_.append(x3)
        #x4 = self.Maxpool(x3)
        x4_1 = self.Conv4(x3)
        x4_2 = self.edgat4(x3)
        x4 = torch.cat((x4_1,x4_2),dim=1)
        x5_1 = self.Conv5(x4)
        x5_2 = self.edgat5(x4)
        x5 = torch.cat((x5_1,x5_2),dim=1)
        #x_.append(x4)
        # x1,x2,x3,x4都返回print 出来一下，resize,channel数有点多，到时候统一分成八个
        #print(x.shape,x1.shape,x2.shape,x3.shape,x4.shape)#[8,3,512,512] [8,32,512,512] [8,64,256,256] [8,128,128,128] [8,256,64,64]
        x6 = self.Conv6(x5)
        x7 = self.Conv7(x6)
        x8 = self.Conv8(x7)
        x9 = self.Conv9(x8)
        x10 = self.Conv10(x9)
        #x6 = self.Conv6(x5)
        #x7 = self.Trans_D(x5)
	#这块应该要对这个x8进行一个归一化的操作
        d1 = self.pool_tofc_D(x10)
        feature_d = d1.squeeze(dim=3).squeeze(dim=2)
        pad = self.liner(feature_d)
        out = pad
        out = self.sig(out)
        return out,x10,feature_d,seg_input,self.liner.weight
    
    
