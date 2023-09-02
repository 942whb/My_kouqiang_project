from PIL import Image
import numpy as np
import scipy
import os
from random import randint
from torch.utils import data
import numpy as np
import os
import torch
from torchvision.transforms import ToPILImage
import cv2
import nibabel as nib
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision import transforms 
import paired_transforms_tv04 as p_tr
# 打开图片
def img_pro(img):
    img = np.array(img)
    img = transforms.ToTensor()(img)
    img = p_tr.Resize((256,256))(img)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img) 
    to_PILimage = transforms.ToPILImage()
    img = to_PILimage(img)  
    return img
def main():
    imgA_path = '/home/whb/kouqiang_classify/data/orignal_data/image_c1/ame (411).jpg'
    imgB_path = '/home/whb/kouqiang_classify/data/ame_/ame_(411).jpg'
    imgA = Image.open(imgA_path)
    imgB = Image.open(imgB_path)
    #imgA = img_pro(imgA)
    #imgB = img_pro(imgB)
    #归一化的操作有没有说法
    #imgA = p_tr.Resize((256,256))(imgA)
    #imgB = p_tr.Resize((256,256))(imgB)
    # 获取目标图片的大小(也就是长和高的像素)
    width, height = imgA.size
    # 获取图片每个像素的颜色
    for x in range(0, width-1):
        for y in range(0, height-1):
            color1 = imgA.getpixel((x, y))
            color2 = imgB.getpixel((x, y))
    # 对比两张图片的像素颜色  相同的地方变白，不同的地方变黑这个地方可以稍微调整一下比如差距小于多少的时候
            if color1 == color2:
    # 改变像素颜色 | 255.255.255为RBG的白色 0.0.0为黑色
                imgA.putpixel((x, y), (0, 0, 0))
            else:
                imgA.putpixel((x, y), (255, 255, 255))
    # 输出对比结果  图片名随意命名
    imgC_path = '/home/whb/kouqiang_classify/data/ameR/ameR(411).jpg'
    imgA.save(imgC_path)
    for i in range(30):
        strA = str(i+11)
        strA_ = str(i+10)
        imgA_path = imgA_path.replace(strA_, strA)
        imgB_path = imgB_path.replace(strA_ , strA)
        imgC_path = imgC_path.replace(strA_ , strA)
        imgA = Image.open(imgA_path)
        imgB = Image.open(imgB_path)
        #imgA = img_pro(imgA)
        #imgB = img_pro(imgB)
        #imgA = resize(imgA)
        #imgB = resize(imgB)
        #归一化的操作有没有说法
        #imgA = p_tr.Resize((256,256))(imgA)
        #imgB = p_tr.Resize((256,256))(imgB)
        # 获取目标图片的大小(也就是长和高的像素)
        width, height = imgA.size
        # 获取图片每个像素的颜色
        for x in range(0, width-1):
            for y in range(0, height-1):
                color1 = imgA.getpixel((x, y))
                color2 = imgB.getpixel((x, y))
        # 对比两张图片的像素颜色  相同的地方变白，不同的地方变黑这个地方可以稍微调整一下比如差距小于多少的时候
                if judge_color(x,y,imgA,imgB):
        # 改变像素颜色 | 255.255.255为RBG的白色 0.0.0为黑色
                    imgA.putpixel((x, y), (255, 255, 255))
                else:
                    imgA.putpixel((x, y), (0, 0, 0))
        # 输出对比结果  图片名随意命名
        imgA.save(imgC_path)

def judge_color(x,y,imgA,imgB):
    color1 = []
    color2 = []
    for i in range(-2,2):
        for j in range(-2,2):
            if(x+i>=0):
                if(y+j>=0):
                    color1.append(imgA.getpixel((x+i,y+j)))
                    color2.append(imgB.getpixel((x+i,y+j)))
                else:
                    color1.append(imgA.getpixel((x+i,0)))
                    color2.append(imgB.getpixel((x+i,0)))
            else:
                if(y+j>=0):
                    color1.append(imgA.getpixel((0,y+j)))
                    color2.append(imgB.getpixel((0,y+j)))

                else:
                    color1.append(imgA.getpixel((0,0)))
                    color2.append(imgB.getpixel((0,0)))
    for i in range (0,len(color1)):
        if(color1[i]==color2[i]): return 1
    return 0               
            

if __name__ == '__main__':
    main()