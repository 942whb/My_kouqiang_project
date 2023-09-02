from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr

import os
import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops
import torch
import os
class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        df = pd.read_csv(csv_path)
        self.image_list = df['image']
        self.labels = df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
        self.transforms = transforms

    def __getitem__(self, index):
        # Load image and labels
        image_path = self.image_list[index]
        base_path = "/data/whb/HAM10000/images/"
        image_path = os.path.join(base_path, image_path + ".jpg")
        image = Image.open(image_path)

        # Apply transformations if specified
        if self.transforms is not None:
            image = self.transforms(image)

        # Convert labels to tensors
        labels = torch.tensor(self.labels[index], dtype=torch.int)

        return image, labels

    def __len__(self):
        return len(self.image_list)




class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size):
        df = pd.read_csv(csv_path)
        self.cl1_list = df.cl1_paths
        self.cl2_list = df.cl2_paths
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # # load image and mask
        img_c1 = Image.open(self.cl1_list[index])
        img_c2 = Image.open(self.cl2_list[index])
        original_sz = img.size[1], img.size[0]  # in numpy convention

        # # load image and mask
        # img = Image.open(self.im_list[index])
        # original_sz = img.size[1], img.size[0]  # in numpy convention
        # mask = Image.open(self.mask_list[index]).convert('L')
        # img, coords_crop = self.crop_to_fov(img, mask)
        # print(self.im_list[index], 'original size inside dataset', original_sz)

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img_c1 = tr(img_c1)  # only transform image
        img_c2 = tr(img_c2)

        return img_c1, img_c2, original_sz, self.cl1_list[index], self.cl2_list[index]

    def __len__(self):
        return len(self.cl1_list)


def build_pseudo_dataset(train_csv_path, test_csv_path, path_to_preds):
    # assumes predictions are in path_to_preds and have the same name as images in the test csv
    # image extension does not matter
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # If there are more pseudo-segmentations than training segmentations
    # we bootstrap training images to get same numbers
    missing = test_df.shape[0] - train_df.shape[0]
    if missing > 0:
        extra_segs = train_df.sample(n=missing, replace=True, random_state=42)
        train_df = pd.concat([train_df, extra_segs])


    train_im_list = list(train_df.im_paths)
    train_gt_list = list(train_df.gt_paths)
    train_mask_list = list(train_df.mask_paths)

    test_im_list = list(test_df.im_paths)
    test_mask_list = list(test_df.mask_paths)

    test_preds = [n for n in os.listdir(path_to_preds) if 'binary' not in n and 'perf' not in n]
    test_pseudo_gt_list = []

    for n in test_im_list:
        im_name_no_extension = n.split('/')[-1][:-4]
        for pred_name in test_preds:
            pred_name_no_extension = pred_name.split('/')[-1][:-4]
            if im_name_no_extension == pred_name_no_extension:
                test_pseudo_gt_list.append(osp.join(path_to_preds, pred_name))
                break
    train_im_list.extend(test_im_list)
    train_gt_list.extend(test_pseudo_gt_list)
    train_mask_list.extend(test_mask_list)
    return train_im_list, train_gt_list, train_mask_list


def get_train_val_datasets(csv_path_train, csv_path_val, tg_size=(512, 512)):

    train_dataset = TrainDataset(csv_path=csv_path_train)
    val_dataset = TrainDataset(csv_path=csv_path_val)
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)#这里就是把整体的尺寸都变成了512*512
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=90, center=(256,256),fill=(0, 0, 0), fill_tg=(0,))#Rotate the image by angle.
    #scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.00))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue =0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize, jitter,rotate,tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    #train_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), num_workers=0):
    train_dataset, val_dataset = get_train_val_datasets(csv_path_train, csv_path_val, tg_size=tg_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_datasets(data_path, csv_path, tg_size=(1024,1024), label_values=(0, 255)):

    test_dataset = TrainDataset(csv_path=csv_path, label_values=label_values)
    
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    
    test_transforms = p_tr.Compose([resize, tensorizer])
    test_dataset.transforms = test_transforms
    
    return test_dataset


