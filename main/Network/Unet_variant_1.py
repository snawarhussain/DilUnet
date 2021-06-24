from abc import ABC

import numpy
import torch
import torch.nn as nn
from PIL import Image

from main.utils.tem import visualize
from .skip_connections import *


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


def crop(tensor, target_tensor):
    tensor_size = tensor.shape[2]
    target_size = target_tensor.shape[2]
    if (abs(tensor_size - target_size) == 1):
        delta = 1
    else:
        delta = abs(tensor_size - target_size) // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UpSampleMultiOutput(nn.Module):
    def __init__(self, scale, input_channels, ):
        super(UpSampleMultiOutput, self).__init__()
        self.scale = scale
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.TransConv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=1,
                                            kernel_size=(1, 1), stride=(1, 1))
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.up(x)
        return self.out(self.TransConv(x))


class MFEBlock(nn.Module):
    def __init__(self, input_channels, dilated_channels, output_channels):
        super(MFEBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_3_3_1 = nn.Sequential(
            nn.Conv2d(input_channels, dilated_channels, kernel_size=(3, 3),
                      padding=(1, 1), dilation=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(dilated_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dilated_channels, output_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)

        )

        self.conv_3_3_2 = nn.Sequential(
            nn.Conv2d(input_channels, dilated_channels,
                      kernel_size=3, padding=2, dilation=2, stride=1),
            nn.BatchNorm2d(dilated_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dilated_channels, output_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)

        )
        self.conv_3_3_3 = nn.Sequential(
            nn.Conv2d(input_channels, dilated_channels,
                      kernel_size=3, padding=3, dilation=3, stride=1),
            nn.BatchNorm2d(dilated_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dilated_channels, output_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)

        )

        # self.conv_1_1 = nn.Conv2d(dilated_channels, output_channels,
        #                           kernel_size=1, stride=1)
        # self.conv_3_3_1 = nn.Conv2d(input_channels, dilated_channels,
        #                             kernel_size=3, padding=1, dilation=1, stride=1)
        # self.conv_3_3_2 = nn.Conv2d(input_channels, dilated_channels, kernel_size=3,
        #                             padding=2, dilation=2)
        # self.conv_3_3_3 = nn.Conv2d(input_channels, dilated_channels,
        #                             kernel_size=3, dilation=3, padding=3, stride=1)
        # self.conv_1_1_1 = nn.Conv2d(input_channels, output_channels,
        #                           kernel_size=1, stride=1)
        #
        # self.batchnorm = nn.BatchNorm2d(dilated_channels)
        # self.batchnorm1 = nn.BatchNorm2d(output_channels)
        # self.act_func = nn.ReLU()

    def forward(self, x):
        # output1 = self.conv_3_3_1(x)
        # output1 = self.batchnorm(output1)
        # output1 = self.act_func(output1)
        # output1 = self.conv_1_1(output1)
        # output1 = self.batchnorm1(output1)
        # output1 = self.act_func(output1)
        # output2 = self.conv_3_3_2(x)
        # output2 = self.act_func(self.batchnorm(output2))
        # output2 = self.conv_1_1(output2)
        # output2 = self.act_func(self.batchnorm1(output2))
        #
        # output3 = self.conv_3_3_3(x)
        # output3 = self.act_func(self.batchnorm(output3))
        # output3 = self.conv_1_1(output3)
        # output3 = self.act_func(self.batchnorm1(output3))
        # out = torch.cat((output1, output2), dim=1)
        # return torch.cat((out, output3), dim=1)
        output1 = self.conv_3_3_1(x)
        output2 = self.conv_3_3_2(x)
        output3 = self.conv_3_3_3(x)
        out = torch.cat((output1, output2), dim=1)
        return torch.cat((out, output3), dim=1)


class DcBlock(nn.Module, ABC):
    def __init__(self, in_c, interm_c, out_c, dilation=1):
        super(DcBlock, self).__init__()
        self.in_c = in_c
        self.interm_c = interm_c
        self.out_c = out_c
        self.dilated_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_c, out_channels=self.interm_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.interm_c, out_channels=self.out_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3,
            #           stride=1, padding=dilation, dilation=dilation),
            # #nn.BatchNorm2d(num_features=self.out_c),
            # nn.ReLU(inplace=True)

        )

    def forward(self, X):
        output = self.dilated_block(X)
        # print(output.shape)
        return output


class DecoderDcBlock(nn.Module, ABC):
    def __init__(self, in_c, interm_c, out_c, dilation=1):
        super(DecoderDcBlock, self).__init__()
        self.in_c = in_c
        self.interm_c = interm_c
        self.out_c = out_c
        self.dilated_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_c, out_channels=self.interm_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.interm_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.interm_c, out_channels=self.out_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3,
            #           stride=1, padding=dilation, dilation=dilation),
            # #nn.BatchNorm2d(num_features=self.out_c),
            # nn.LeakyReLU(inplace=True)

        )

    def forward(self, X):
        output = self.dilated_block(X)
        # print(output.shape)
        return output


class DcBlockLast(nn.Module, ABC):
    def __init__(self, in_c, interm_c, out_c, dilation=1):
        super(DcBlockLast, self).__init__()
        self.in_c = in_c
        self.interm_c = interm_c
        self.out_c = out_c
        self.dilated_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_c, out_channels=self.interm_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.interm_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.interm_c, out_channels=self.out_c, kernel_size=3,
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3,
            #           stride=1, padding=dilation, dilation=dilation),
            # nn.LeakyReLU(inplace=True)
            # nn.BatchNorm2d(num_features=self.out_c)

        )

    def forward(self, X):
        output = self.dilated_block(X)
        # print(output.shape)
        return output


class UnetVariant_1(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(UnetVariant_1, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout = nn.Dropout2d(0.1)
        self.MFEBlock1 = MFEBlock(input_channels, 16, 2)
        self.MFEBlock2 = MFEBlock(input_channels, 32, 4)
        self.MFEBlock3 = MFEBlock(input_channels, 64, 6)
        self.MFEBlock4 = MFEBlock(input_channels, 128, 8)

        self.MultiOutput1 = UpSampleMultiOutput(8, 256)
        self.MultiOutput2 = UpSampleMultiOutput(4, 128)
        self.MultiOutput3 = UpSampleMultiOutput(2, 64)

        self.DcBlock1 = DcBlock(6, 16, 32)
        self.skipPath1 = SkipConnection01(32, 32)

        self.DcBlock2 = DcBlock(32 + 12, 48, 64)
        self.skipPath2 = SkipConnection02(64, 64)

        self.DcBlock3 = DcBlock(64 + 18, 96, 128, dilation=2)
        self.skipPath3 = SkipConnection03(128, 128)

        self.DcBlock4 = DcBlock(128 + 24, 192, 256, dilation=3)
        self.skipPath4 = SkipConnection04(256, 256)

        self.DcBlock5 = DcBlock(256, 384, 512)
        # r Decoder part
        self.ConvTrans1 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.UpDcBlock1 = DecoderDcBlock(512, 384, 256, dilation=2)

        self.ConvTrans2 = nn.ConvTranspose2d(256, 128,  kernel_size=(2, 2), stride=(2, 2))
        # self.UpDcBlock2 = DecoderDcBlock(384, 192, 128)
        self.UpDcBlock2 = DecoderDcBlock(256, 192, 128, dilation=1)

        self.ConvTrans3 = nn.ConvTranspose2d(128, 64,  kernel_size=(2, 2), stride=(2, 2))
        # self.UpDcBlock3 = DecoderDcBlock(192, 96, 64)
        self.UpDcBlock3 = DecoderDcBlock(128, 96, 64, dilation=1)

        self.ConvTrans4 = nn.ConvTranspose2d(64, 32,  kernel_size=(2, 2), stride=(2, 2))
        self.UpDcBlock4 = DecoderDcBlock(64, 48, 32, dilation=3)

        self.output = nn.ConvTranspose2d(32, out_channels=self.output_channels, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.SE = SELayer(4)
        self.output_final = nn.ConvTranspose2d(4, out_channels=self.output_channels, kernel_size=1)

    def forward(self, X):
        # downsample image
        image1 = X
        image2 = nn.functional.max_pool2d(image1, kernel_size=2, stride=2)
        image3 = nn.functional.max_pool2d(image1, kernel_size=4, stride=4)
        image4 = nn.functional.max_pool2d(image1, kernel_size=8, stride=8)
        # image4 = nn.functional.max_pool2d(image3, kernel_size=2, stride=2)

        image1 = self.MFEBlock1(image1)
        image2 = self.MFEBlock2(image2)
        image3 = self.MFEBlock3(image3)
        image4 = self.MFEBlock4(image4)
        # photo = image1.cpu().detach().numpy()
        # visualize(photo[0][0])
        X1Block = self.DcBlock1(image1)

        # X1 = self.dropout(X1)
        X1BlockDown = nn.functional.max_pool2d(X1Block, kernel_size=2, stride=2)
        X2 = torch.cat((image2, X1BlockDown), dim=1)
        X2Block = self.DcBlock2(X2)

        X2BlockDown = nn.functional.max_pool2d(X2Block, kernel_size=2, stride=2)

        X3 = torch.cat((image3, X2BlockDown), dim=1)
        X3Block = self.DcBlock3(X3)

        X3BlockDown = nn.functional.max_pool2d(X3Block, kernel_size=2, stride=2)

        X4 = torch.cat((image4, X3BlockDown), dim=1)

        X4Block = self.DcBlock4(X4)
        X4BlockDown = nn.functional.max_pool2d(X4Block, kernel_size=2, stride=2)
        # X2 = self.dropout(X2)
        X5Block = self.DcBlock5(X4BlockDown)

        # X5Block = self.up(X5Block)
        X5BlockUp = self.ConvTrans1(X5Block)

        X4BlockSkip = self.skipPath4(X4Block)
        X3BlockSkip = self.skipPath3(X3Block)
        X2BlockSkip = self.skipPath2(X2Block)
        X1BlockSkip = self.skipPath1(X1Block)
        # X3 = self.dropout(X3)
        #X5BlockUp = nn.functional.interpolate(X5BlockUp, X4BlockSkip.shape[2])
        X6 = torch.cat((X4BlockSkip, X5BlockUp), dim=1)
        X6Block = self.UpDcBlock1(X6)

        # X6Block = self.up(X6Block)
        X6BlockUp = self.ConvTrans2(X6Block)

        #X6BlockUp = nn.functional.interpolate(X6BlockUp, X3BlockSkip.shape[2])
        X7 = torch.cat((X3BlockSkip, X6BlockUp), dim=1)

        X7Block = self.UpDcBlock2(X7)

        # X7Block = self.up(X7Block)
        X7BlockUp = self.ConvTrans3(X7Block)
        #X7BlockUp = nn.functional.interpolate(X7BlockUp, X2BlockSkip.shape[2])
        X8 = torch.cat((X2BlockSkip, X7BlockUp), dim=1)

        X8Block = self.UpDcBlock3(X8)

        # X8Block = self.up(X8Block)
        X8BlockUp = self.ConvTrans4(X8Block)
        #X8BlockUp = nn.functional.interpolate(X8BlockUp, X1BlockSkip.shape[2])
        X9 = torch.cat((X1BlockSkip, X8BlockUp), dim=1)
        X9Block = self.UpDcBlock4(X9)

        MultiOutput1 = nn.functional.interpolate(self.MultiOutput1(X6Block), 512)
        MultiOutput2 = nn.functional.interpolate(self.MultiOutput2(X7Block), 512)
        MultiOutput3 = nn.functional.interpolate(self.MultiOutput3(X8Block), 512)
        MultiOutput4 = self.sig(self.output(X9Block))
        out_stacked = torch.cat((.1*MultiOutput1, .2*MultiOutput2,
                                      0.3*MultiOutput3, MultiOutput4), dim=1)
        # img1 =numpy.uint8(MultiOutput1.cpu().detach().numpy().squeeze()*255)
        # img1 = Image.fromarray(img1)
        # img1.save('multioutput1.jpg')
        # img2 = Image.fromarray(numpy.uint8(MultiOutput2.cpu().detach().numpy().squeeze()*255))
        # img2.save('multioutput2.jpg')
        # img3 = Image.fromarray(numpy.uint8(MultiOutput3.cpu().detach().numpy().squeeze()*255))
        # img3.save('multioutput3.jpg')
        # out_stacked = torch.cat((0.001*MultiOutput1, MultiOutput2,
        #                          MultiOutput3, MultiOutput4), dim=1)
        #
        # se_ouput = self.SE(out_stacked)
        out = self.output_final(out_stacked)
        return self.sig(out)

# if __name__ == "__main__":
#     x = torch.randn((2, 1, 584, 584))
#     model = UnetVariant_1(1, 1)
#     print(model)
#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)
