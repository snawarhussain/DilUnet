from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score, auc


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


AUC=[]
whole_Acc=[]
whole_IOU=[]
whole_sensitivity=[]
whole_specificity=[]
from sklearn.metrics import roc_auc_score
import glob