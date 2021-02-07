# encoding: utf-8
import torch.utils.data as data
from PIL import Image
import os
import torch
import numpy as np
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import glob

def make_dataset_DRIVE():
    rootPath='data/train/'
    path_imgs = glob.glob(rootPath + 'image/*.jpg')
    path_imgs = sorted(path_imgs)
    imgs = []
    for path_img in path_imgs[:]:
        name = os.path.basename(path_img)
        img=path_img
        mask=os.path.join(rootPath+'mask',name)
        imgs.append((img,mask))
    return imgs

class LiverDataset(data.Dataset):
    def __init__(self, size, transform=None, target_transform=None):
        imgs = make_dataset_DRIVE()
        self.size = size
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        size=self.size
        x_path, mask_path= self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((size,size),Image.ANTIALIAS)
        img_mask = Image.open(mask_path)
        #img_mask = img_mask.resize((512,512),Image.ANTIALIAS)
        img_mask = np.asarray(img_mask).astype(np.uint8)
        size = (int(512), int(512))
        img_mask = cv2.resize(img_mask, size, interpolation=cv2.INTER_LINEAR)
        ret, img_mask = cv2.threshold(img_mask, 105, 255, cv2.THRESH_BINARY)
        img_mask = np.expand_dims(np.array(img_mask, np.float32), -1)
        #img_mask = np.expand_dims(np.array(img_mask, np.float32), -1)
        #print('img_mask1', img_mask.shape)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
            img_mask=img_mask/255.
            #img_mask = BW_img1(img_mask, 0.3)
            #print('img_mask',img_mask.shape)
            return img_x, img_mask
    def __len__(self):
         return len(self.imgs)

def make_dataset_val():
    imgs=[]
    root1="data/origin_test_1st/image"
    root2="data/origin_test_1st/mask"
    for i in range(20):
        i=i+1
        img=os.path.join(root1,"%02d.jpg"%i)
        maskod=os.path.join(root2,"%02d.jpg"%i)
        imgs.append((img,maskod))
    return imgs

def BW_img1(input, thresholding=0.5):
    input[input >= thresholding]=1
    input[input < thresholding]=0
    return input

import scipy
from skimage.measure import label, regionprops
def BW_img(input, thresholding):
    binary = input > thresholding
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max+1] = 0
    return np.array(scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int)),np.float32)

class LiverDataset_val(data.Dataset):
    def __init__(self, size, transform=None, target_transform=None):
        imgs = make_dataset_val()
        self.size = size
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, mask_path= self.imgs[index]
        size=self.size
        img_x = Image.open(x_path)
        img_x = img_x.resize((size,size),Image.ANTIALIAS)
        img_mask = Image.open(mask_path)
        #img_mask = img_mask.resize((512,512),Image.ANTIALIAS)
        #############
        img_mask = np.asarray(img_mask).astype(np.uint8)
        size = (int(512), int(512))
        img_mask = cv2.resize(img_mask, size, interpolation=cv2.INTER_LINEAR)
        ret, img_mask = cv2.threshold(img_mask, 105, 255, cv2.THRESH_BINARY)
        #############
        img_mask = np.expand_dims(np.array(img_mask, np.float32), -1)
        #print('img_mask1', img_mask.shape)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
            img_mask=img_mask/255.
            #img_mask = BW_img1(img_mask, 0.3)
            #print('img_mask',img_mask.shape)
            return img_x, img_mask
    def __len__(self):
        return len(self.imgs)