# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
import torch
"""
"""

class PreBlock(nn.Module):
    def __init__(self, in_channels, out_channels,act_func=nn.ReLU(inplace=True)):
        super(PreBlock, self).__init__()
        out8=int(out_channels/8)
        out4=int(out_channels/4)
        out2=int(out_channels/2)

        self.conv1_1 = nn.Conv2d(in_channels, out2, 3,padding=1)
        self.conv2_1 = nn.Conv2d(in_channels, out8, 3,padding=1)
        self.conv2_2 = nn.Conv2d(out8, out8, 1)
        self.conv3_1 = nn.Conv2d(in_channels, out8, 5, padding=2)
        self.conv4_1 = nn.Conv2d(in_channels, out8, 3, stride=1,dilation=3, padding=3)
        self.conv5_2 = nn.Conv2d(out8, out8, 3, stride=1,dilation=5, padding=5)

        self.act_func = act_func
        self.bn4 = nn.BatchNorm2d(out8)
        self.bn16 = nn.BatchNorm2d(out2)

    def forward(self, x):
        out1_1 = self.conv1_1(x)
        out1_1 = self.bn16(out1_1)
        out1_1 = self.act_func(out1_1)  # 3×3 16

        out2_1 = self.conv2_1(x)
        out2_1 = self.bn4(out2_1)
        out2_1 = self.act_func(out2_1)
        out2_2 = self.conv2_2(out2_1)
        out2_2 = self.bn4(out2_2)
        out2_2 = self.act_func(out2_2)

        out3_1 = self.conv3_1(x)
        out3_1 = self.bn4(out3_1)
        out3_1 = self.act_func(out3_1)

        out4_1=self.conv4_1(x)  # 3×3 d=3
        out4_2 = self.bn4(out4_1)
        out4_2 = self.act_func(out4_2)

        out5_1=self.conv4_1(x)  # 3×3 d=3
        out5_1 = self.bn4(out5_1)
        out5_1 = self.act_func(out5_1)
        out5_2=self.conv5_2(out5_1)  # 3×3 d=3
        out5_2 = self.bn4(out5_2)
        out5_2 = self.act_func(out5_2)

        out = torch.cat([out1_1, out2_2], dim=1)
        out = torch.cat([out, out3_1], dim=1)
        out = torch.cat([out, out4_2,out5_2], dim=1)
        return out
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,act_func=nn.ReLU(inplace=True),dilation=1):
        super(VGGBlock, self).__init__()
        self.dilation=dilation
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, dilation=dilation,padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, dilation=dilation,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Conv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv1(input)

class Uplus_CP(nn.Module):
    def __init__(self, args):
        # super().__init__() #python3.5
        super(Uplus_CP,self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(4, 4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convA1 = PreBlock(1,nb_filter[1])
        self.convB = PreBlock(1,nb_filter[0])
        self.convC = PreBlock(1,nb_filter[1])
        self.convD = PreBlock(1,nb_filter[2])

        self.conv0_01 = VGGBlock(nb_filter[1],nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[1],nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[2],nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[3],nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock1(nb_filter[4],nb_filter[4], nb_filter[4])

        self.conv4_a1 = nn.Conv2d(448, nb_filter[2], 3, dilation=1,padding=1)
        self.conv4_b = nn.Conv2d(nb_filter[2], nb_filter[1], 3, dilation=3,padding=3)
        self.conv4_c1 = nn.Conv2d(nb_filter[1], nb_filter[1], 3, dilation=5,padding=5)

        self.SELayer_3 = SELayer(nb_filter[3]+nb_filter[3], reduction=16)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.SELayer_2 = SELayer(nb_filter[2]+nb_filter[3], reduction=16)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.SELayer_1 =SELayer(nb_filter[1]+nb_filter[2], reduction=16)
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.SELayer_0 = SELayer(nb_filter[0]+nb_filter[1], reduction=16)
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.F = nn.Conv2d(32, args.output_channels, 3,dilation=1,padding=1)

    def forward(self, input):

        CPA = input
        CPB = self.pool(CPA)
        CPC = self.pool(CPB)
        CPD = self.pool(CPC)

        A_0 = self.convA1(CPA) # 32
        B_0 = self.convB(CPB) # 32
        C_0 = self.convC(CPC) # 64
        D_0 = self.convD(CPD) # 128

        x0_0 = self.conv0_01(A_0) #32
        cp_1 = torch.cat([B_0 , self.pool(x0_0)], dim=1) #64

        x1_0 = self.conv1_0(cp_1)
        cp_2 = torch.cat([C_0, self.pool(x1_0)], dim=1) #128

        x2_0 = self.conv2_0(cp_2)
        cp_3=torch.cat([D_0, self.pool(x2_0)], dim=1) #256

        x3_0 = self.conv3_0(cp_3) # 256


        DC1=self.pool1(cp_1)
        DC2=self.pool(cp_2)
        DC3=cp_3
        DC=torch.cat([DC1,DC2,DC3], 1)  #448

        x4_0 = self.conv4_a1(DC) #  128
        x4_1 = self.conv4_b(x4_0) #  64
        x4_2 = self.conv4_c1(x4_1) #  64
        x4_3=torch.cat([x4_0,x4_1,x4_2], 1)#256

        x3_0 = nn.functional.interpolate(x3_0, x4_3.shape[2])
        x3_1 = self.conv3_1(self.SELayer_3(torch.cat([(x3_0), (x4_3)], 1)))

        x2_0 = nn.functional.interpolate(x2_0, self.up(x3_1).shape[2])
        x2_1 = self.conv2_1(self.SELayer_2(torch.cat([(x2_0), self.up(x3_1)], 1)))

        x1_0 = nn.functional.interpolate(x1_0, self.up(x2_1).shape[2])
        x1_2 = self.conv1_2(self.SELayer_1(torch.cat([(x1_0), self.up(x2_1)], 1)))

        x0_0 = nn.functional.interpolate(x0_0, self.up(x1_2).shape[2])
        x0_3 = self.conv0_3(self.SELayer_0(torch.cat([(x0_0), self.up(x1_2)], 1))  )#
        x0_4=self.conv0_4(x0_3)
        output = self.F(x0_4)
        output = nn.Sigmoid()(output)
        return output