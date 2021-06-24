# encoding: utf-8
"""
直接LiverDataset 从 dataloader 中一张一张读取图像 图像的格式是IMAGE格式的
不保存，直接计算IOU等
"""
import os
from sklearn.metrics import roc_curve, roc_auc_score, auc
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch.utils.data as data
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
import cv2

def make_dataset(path_img):
    name=os.path.basename(path_img)
    newpath=os.path.dirname(path_img)
    newpath=newpath.replace('image','mask')
    imgs=[]
    img=path_img
    mask=newpath+'/'+name
    imgs.append((img,mask))
    return imgs
def make_datasetcrop(path_img):
    name=os.path.basename(path_img)
    newpath=os.path.dirname(path_img)
    newpath=newpath.replace('image','mask')
    imgs=[]
    img=path_img
    mask=newpath+'/'+name
    imgs.append((img,mask))
    return imgs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--deepsupervision', default=False)

    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--output-channels', default=1, type=int,
                        help='output channels')
    args = parser.parse_args()

    return args

class LiverDataset(data.Dataset):
    def __init__(self, path_img,size, transform=None, target_transform=None):
        imgs = make_dataset(path_img)
        self.size=size
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, mask_path= self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((self.size, self.size),Image.ANTIALIAS)
        img_mask = Image.open(mask_path)
        #img_mask = img_mask.resize((256, 256),Image.ANTIALIAS)
        img_mask = np.asarray(img_mask).astype(np.uint8)
        size = (int(512), int(512))
        img_mask = cv2.resize(img_mask, size, interpolation=cv2.INTER_LINEAR)
        ret, img_mask = cv2.threshold(img_mask, 105, 255, cv2.THRESH_BINARY)
        img_mask = np.expand_dims(np.array(img_mask, np.float32), -1)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            #img_mask = self.target_transform(img_mask)
            img_mask=img_mask/255
            img_mask = self.target_transform(img_mask)
            #img_mask=BW_img(img_mask,0.3)
            return img_x, img_mask
    def __len__(self):
         return len(self.imgs)

class LiverDatasetcrop(data.Dataset):
    def __init__(self, path_img,size, transform=None, target_transform=None):
        imgs = make_datasetcrop(path_img)
        self.size=size
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, mask_path= self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((self.size, self.size),Image.ANTIALIAS)
        img_mask = Image.open(mask_path)
        #img_mask = img_mask.resize((256, 256),Image.ANTIALIAS)
        img_mask = np.asarray(img_mask).astype(np.uint8)
        size = (int(256), int(256))
        img_mask = cv2.resize(img_mask, size, interpolation=cv2.INTER_LINEAR)
        ret, img_mask = cv2.threshold(img_mask, 105, 255, cv2.THRESH_BINARY)
        img_mask = np.expand_dims(np.array(img_mask, np.float32), -1)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            #img_mask = self.target_transform(img_mask)
            img_mask=img_mask/255
            img_mask = self.target_transform(img_mask)
            #img_mask=BW_img(img_mask,0.3)
            return img_x, img_mask
    def __len__(self):
         return len(self.imgs)
def BW_img(image, thresholding):
    """二值化分割后的图像"""
    image[image >= thresholding] = 1
    image[image < thresholding] = 0  #
    return image

# 评价分割精度 几个指标 1.iou 2.Acc 3.CDR误差
#1.iou

def getIOU(output,target):
    """ 单张图片 获得IOU：重叠评分S"""
    smooth=0.00001
    inse = sum(sum( (output*target)))  # AND
    image=output+target
    binary = image >= 1
    binary1 = image < 1
    image[binary] = 1;
    image[binary1] = 0;
    union = sum(sum( (image)))
    iou = (inse + smooth) / (union + smooth)
    return iou

def getAcc(output,target):
    TP = sum(sum((output*target)))  # AND
    output1=np.where(output<1,1,0)# output中01互换
    Positivebing=output1 * target
    FN=sum(sum( (Positivebing)))# 计算
    sensitivity=TP/(TP+FN)
    target1=np.where(target<1,1,0)# output中01互换
    negbing=output1*target1
    TN=sum(sum( (negbing)))
    FP=sum(sum( (output*target1)))
    specificity=TN/(TN+FP)
    Acc=(TP+TN)/(TP+FN+TN+FP)
    #Acc=0.5*(sensitivity+specificity)
    return Acc,sensitivity,specificity

def get_aera_and_ratio(mask_disc, mask_cup):
    '''返回视杯视盘和盘沿面积，还有他们之间的占比 '''
    disc_area = np.sum(mask_disc)
    cup_area = np.sum(mask_cup)
    rim_area = disc_area - cup_area
    CD_area_ratio = round(cup_area / disc_area, 2)
    rim_to_disc_ratio = 1 - CD_area_ratio
    return disc_area, cup_area, rim_area, CD_area_ratio, rim_to_disc_ratio

# def U_net():
#     from U_net import UNet
#     model = UNet(3,1).to(device)
#     #model = model.cuda()
#     #model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load('DRIVE_1st_Unet_3.pkl') )
#     return model
# def MESNET():
#     #from modify_UP import Uplus_CP
#     from MES_net import mesnet
#     args = parse_args()
#     model = mesnet(args).to(device)
#     #model = model.cuda()
#     model.load_state_dict(torch.load('DRIVE_1st_PSA_AUC0.9853.pkl'))
#     return model


def get_roc_curve(y_true, y_prob, average='samples'):
    '''
    绘制ROC曲线并计算出AUC
    :param y_true: 真实类别
    :param y_prob: 类别为1的预测概率
    :return: None
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_prob)
    _auc = roc_auc_score(y_true, y_prob, average=average)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % _auc)  # 生成ROC曲线
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.show()

# model = MESNET()
AUC=[]
whole_Acc=[]
whole_IOU=[]
whole_sensitivity=[]
whole_specificity=[]
from sklearn.metrics import roc_auc_score
import glob
#ROI_root='data/crop/test/image'

ROI_root='data/origin_test_1st/image'
#ROI_root='CHASE_1st/test_tmp/image'
path_imgs = glob.glob(ROI_root + '/*.jpg')
path_imgs = sorted(path_imgs)
num=len(path_imgs)
#imgs_path = 'BinRushedcorrected\BinRushed\mean_masks\OD'
Mes = []
Ori = []
for path_img in path_imgs[0:num]:
    print(path_img)
    name = os.path.basename(path_img)
    size=512
    code=name[:-4]
    liver_dataset = LiverDataset(path_img,size, transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x,target in dataloaders:
            x=x.to(device)
            model.eval()
            y=model(x)
            img_y=torch.squeeze(y).cpu().numpy()

            target=torch.squeeze(target).cpu().numpy()
            #target=BW_img(target,0.5)
            Predimage=img_y
            Predimage=BW_img(img_y, 0.5)
            #from sklearn import metrics
            #fpr, tpr, thresholds = metrics.roc_curve(target.ravel(), Predimage.ravel())
            #get_roc_curve(target.ravel(), Predimage.ravel(), average='macro')
            auc=roc_auc_score(target.ravel(), Predimage.ravel(),average='weighted')
            print('auc=', auc)
            #cv2.imwrite("pred/pred_{}.jpg".format(code), Predimage*255)
            #Predimage.save("pred/pred_{}.jpg".format(code))
            IOU = getIOU(Predimage, target)
            Acc,sensitivity,specificity = getAcc(Predimage, target)
            print('Acc=', Acc)
            print('sensitivity=', sensitivity)
            print('specificity=', specificity)
    Ori.append(target)
    Mes.append(Predimage)
    AUC.append(auc)
    whole_IOU.append(IOU)
    whole_Acc.append(Acc)
    whole_sensitivity.append(sensitivity)
    whole_specificity.append(specificity)
auc = roc_auc_score(np.array(Ori).ravel(), np.array(Mes).ravel(), average='weighted')
get_roc_curve(np.array(Ori).ravel(), np.array(Mes).ravel(), average='macro')

import scipy.io
#scipy.io.savemat('MesDrive9845.mat', {'target': Ori, 'Predimage': Mes})  # 写入mat文件
mean_AUC=sum(AUC)/num
mean_IOU= sum(whole_IOU)/num
mean_Acc=sum(whole_Acc)/num
mean_sensitivity= sum(whole_sensitivity)/num
mean_specificity=sum(whole_specificity)/num
print('mean_IOU',mean_IOU)
print('mean_Acc',mean_Acc)
print('mean_sensitivity',mean_sensitivity)
print('mean_specificity',mean_specificity)
print('mean_AUC',mean_AUC)
