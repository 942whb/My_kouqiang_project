import sys
from .Trans_model_cllasify14 import U_Net as unet
from .AGMB_Transformer import AGGT,AGGTNeckC
from .BYOL import BYOL_res
import torch
from .mymodel import My_Model
from .mymodel_seg import my_model_seg
from .mymodel_new import My_Model_end
from .mymodel_final import My_Model_final
# from .res_unet_adrian import WNet as wnet
# class wnet(torch.nn.Module):
#     def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train'):
#         super(wnet, self).__init__()
#         self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
#         self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
#         self.n_classes = n_classes
#         self.mode=mode

#     def forward(self, x):
#         x1 = self.unet1(x)
#         x2 = self.unet2(torch.cat([x, x1], dim=1))
#         if self.mode!='train':
#             return x2
#         return x1,x2


def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'unet':
        model = unet(in_c=in_c,n_classes=1)
    elif model_name == 'AGMB':
        model = AGGT(AGGTNeckC, [3, 4, 6, 3],n_classes=1)#这个格式不对，其他应该还会，一个一个print试试
    elif model_name == 'BYOL':
        model = BYOL_res(n_classes=1)
    elif model_name == 'MY':
        model = My_Model(in_c=in_c,n_classes=1)   
    #     model = unet(in_c=in_c, n_classes=n_classes, layers=[12,24,48], conv_bridge=True, shortcut=True)
    # elif model_name == 'DHI-GAN':
    #     model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    # elif model_name == 'big_wnet':
    #     model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32,64], conv_bridge=True, shortcut=True)


    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model
# if __name__ == '__main__':
#     import time
#     batch_size = 1
#     batch = torch.zeros([batch_size, 1, 80, 80], dtype=torch.float32)
#     model = get_arch('unet')
#     print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
#     print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
#     start_time = time.time()
#     logits = model(batch)
#     print("--- %s seconds ---" % (time.time() - start_time))

