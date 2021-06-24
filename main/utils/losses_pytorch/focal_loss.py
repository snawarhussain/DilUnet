import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=.25, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss01, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

import torch
import torch.nn as nn

class BCEFocalLoss_cup(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCEFocalLoss(torch.nn.Module):

    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class FocalLoss01(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss01, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCELoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]
        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    z = torch.mul(y_true, y_pred)
    intersection=sum(sum(sum(z)))
    w = (2. * intersection + smooth) / (sum(sum(sum(y_true))) +sum(sum(sum(y_pred))) + smooth)
    return w

def dice_coef2(outputs,labels):
    score0 = dice_coef(y_true=labels[:,0,:,:], y_pred=outputs[:,0,:,:])
    #score1 = dice_coef(y_true=labels[:,1,:,:], y_pred=outputs[:,1,:,:])
    #score = 0.5*(score0+score1)
    return score0

def dice_coef_loss(outputs, labels):
    return 1 - dice_coef2(outputs, labels)

def focal_dice_loss(outputs,labels):
    loss=BCEFocalLoss()
    #loss=nn.BCELoss()
    z=loss(outputs,labels)
    #print("cuploss：{name}, dicsloss: {url}".format(name=cuploss, url=dicsloss))
    #z = 0.5*loss(outputs[:, 0, :, :],labels[:, 0, :, :])+0.5*closs(outputs[:, 1, :, :],labels[:, 1, :, :])
    w = dice_coef_loss(outputs,labels)
    #print("focalloss：{name}, diceloss: {url}".format(name=z, url=w))
    return z+w

def crossentropy_dice_loss1(outputs,labels):
    w = dice_coef2(outputs,labels)
    return w

class FDLoss(nn.Module):
    def __init__(self):
        super(FDLoss, self).__init__()
    def forward(self,outputs,labels):
        return focal_dice_loss(outputs,labels)
