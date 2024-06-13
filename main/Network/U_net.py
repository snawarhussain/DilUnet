from abc import ABC
from torchvision import transforms

import torch
from torch import functional as F
import torch.nn as nn


def double_conv(c_in, c_out):
    conv = nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3,),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3),
        nn.ReLU(inplace=True)

    )
    return conv

def crop(tensor, target_tensor):
    tensor_size = tensor.shape[2]
    target_size = target_tensor.shape[2]
    delta = abs(tensor_size-target_size)//2
    return( tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta])


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.dropOut = nn.Dropout(0.25)

        self.Conv1 = double_conv(1, 32)
        self.Conv2 = double_conv(32, 64)
        self.Conv3 = double_conv(64, 128)
        self.Conv4 = double_conv(128, 256)
        self.Conv5 = double_conv(256, 512)

        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ConvTrans1 = nn.ConvTranspose2d(512,256, kernel_size=2,stride=2)
        self.upConv1 = double_conv(512, 256)


        self.ConvTrans2 = nn.ConvTranspose2d(256,128, kernel_size=2,stride=2)
        self.upConv2 = double_conv(256,128)

        self.ConvTrans3 = nn.ConvTranspose2d(128,64, kernel_size=2,stride=2)
        self.upConv3 = double_conv(128,64)

        self.ConvTrans4 = nn.ConvTranspose2d(64,32, kernel_size=2,stride=2)
        self.upConv4 = double_conv(64,32)

        self.output = nn.ConvTranspose2d(32,1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        r"""Encoder"""
        x1 = self.Conv1(x)##
       # print(x1.shape)
        x2 = self.maxPooling(x1)
       # x2 = self.dropOut(x2)

        x3 = self.Conv2(x2)##
        x4 = self.maxPooling(x3)
      #  x4 = self.dropOut(x4)

        x5 = self.Conv3(x4)##
        x6 = self.maxPooling(x5)
      #  x6 = self.dropOut(x6)

        x7 = self.Conv4(x6)  ##
        x8 = self.maxPooling(x7)
      #  x8 = self.dropOut(x8)

        x9 = self.Conv5(x8)
       # print(x9.shape)

        r"""Decoder"""

        x = self.ConvTrans1(x9)
        y =   nn.functional.interpolate(x7,x.shape[2])
        x = self.upConv1(torch.cat((y,x),dim=1))

        x = self.ConvTrans2(x)
        y = crop(x5,x)
        x = self.upConv2(torch.cat((y,x), dim=1))

        x = self.ConvTrans3(x)
        y = crop(x3, x)
        x = self.upConv3(torch.cat((y, x), dim=1))

        x = self.ConvTrans4(x)
        y = crop(x1, x)
        x = self.upConv4(torch.cat((y, x), dim=1))

        x = self.output(x)
        x = self.sig(x)
       # print(x.shape)

        return x


# if __name__ == "__main__":
#     x = torch.rand((2, 1, 572, 572))
#     model = UNet()
#     print(model)
#     model(x)
