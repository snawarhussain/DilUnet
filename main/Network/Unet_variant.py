from abc import ABC
import torch
import torch.nn as nn

from main.utils.tem import visualize
from .skip_connections import *

def crop(tensor, target_tensor):
    tensor_size = tensor.shape[2]
    target_size = target_tensor.shape[2]
    if (abs(tensor_size - target_size) == 1):
        delta = 1
    else:
        delta = abs(tensor_size - target_size) // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


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
    def __init__(self, in_c, interm_c, out_c, dilation = 1):
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
    def __init__(self, in_c, interm_c, out_c, dilation =1):
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
            #nn.BatchNorm2d(num_features=self.out_c)

        )



    def forward(self, X):
        output = self.dilated_block(X)
       # print(output.shape)
        return output


class UnetVariant(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(UnetVariant, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout = nn.Dropout2d(0.1)
        self.DcBlock1 = DcBlock(input_channels, 16, 32)
        self.skipPath1 = SkipConnection01(32, 32)

        self.DcBlock2 = DcBlock(33, 48, 64)
        self.skipPath2 = SkipConnection02(64, 64)

        self.DcBlock3 = DcBlock(65, 96, 128,dilation=2)
        self.skipPath3 = SkipConnection03(128, 128)

        self.DcBlock4 = DcBlock(129, 192, 256, dilation=3)
        self.skipPath4 = SkipConnection04(256, 256)


        self.DcBlock5 = DcBlock(257, 384, 512)
        # r Decoder part
        self.ConvTrans1 = nn.ConvTranspose2d(512, 256, kernel_size=3,
                                             dilation=2, stride=1, padding=1 )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.UpDcBlock1 = DecoderDcBlock(512, 384, 256,dilation=2)

        self.ConvTrans2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1 )
        #self.UpDcBlock2 = DecoderDcBlock(384, 192, 128)
        self.UpDcBlock2 = DecoderDcBlock(256, 192, 128,dilation=1)

        self.ConvTrans3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1 )
        # self.UpDcBlock3 = DecoderDcBlock(192, 96, 64)
        self.UpDcBlock3 = DecoderDcBlock(128, 96, 64,dilation=1)

        self.ConvTrans4 = nn.ConvTranspose2d(64, 32, kernel_size=3,
                                             dilation=3, stride=1, padding=1 )
        self.UpDcBlock4 = DecoderDcBlock(64, 48, 32, dilation=3)

        self.output = nn.ConvTranspose2d(32, out_channels=self.output_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        #downsample image
        image = X
        image1 = nn.functional.max_pool2d(image, kernel_size=2, stride=2)
        image2 = nn.functional.max_pool2d(image1, kernel_size=2, stride=2)
        image3 = nn.functional.max_pool2d(image2, kernel_size=2, stride=2)
        image4 = nn.functional.max_pool2d(image3, kernel_size=2, stride=2)
        # photo = image.cpu().detach().numpy()
        # visualize(photo[0][0])

        X1_1 = self.DcBlock1(X)

        # X1 = self.dropout(X1)
        X1 = nn.functional.max_pool2d(X1_1, kernel_size=2, stride=2)
        X1 = torch.cat((X1, image1), dim=1)

        X2_1 = self.DcBlock2(X1)

        # X2 = self.dropout(X2)
        X2 = nn.functional.max_pool2d(X2_1, kernel_size=2, stride=2)
        X2 = torch.cat((X2, image2), dim=1)

        X3_1 = self.DcBlock3(X2)

        # X3 = self.dropout(X3)
        X3 = nn.functional.max_pool2d(X3_1, kernel_size=2, stride=2)
        X3 = torch.cat((X3, image3), dim=1)
        X4_1 = self.DcBlock4(X3)

        # X4 = self.dropout(X4)
        X4 = nn.functional.max_pool2d(X4_1, kernel_size=2, stride=2)
        X4 = torch.cat((X4, image4), dim=1)
        X5 = self.DcBlock5(X4)
        # image5 = nn.functional.max_pool2d(image4, kernel_size=2, stride=2)
        # X5 = torch.cat((X5, image5), dim=1)
        # X5 = self.dropout(X5)
        # Decoder part

        X5 = self.up(X5)
        X5 = self.ConvTrans1(X5)
        X5 = nn.functional.interpolate(X5, X4_1.shape[2])
        X4_1 = self.skipPath4(X4_1)
        X6 = torch.cat((X4_1, X5), dim=1)
        X6 = self.UpDcBlock1(X6)
        # X6 = self.dropout(X6)

        X6 = nn.functional.interpolate(X6, X3_1.shape[2])
        X6 = self.up(X6)
        X6 = self.ConvTrans2(X6)
        X3_1 = self.skipPath3(X3_1)
        X3_1 = nn.functional.interpolate(X3_1, X6.shape[2])
        X7 = torch.cat((X3_1, X6), dim=1)
        X7 = self.UpDcBlock2(X7)
        # X7 = self.dropout(X7)

        #
        #
        X7 = self.up(X7)
        X7 = self.ConvTrans3(X7)

        X2_1 = self.skipPath2(X2_1)
        X7 = nn.functional.interpolate(X7, X2_1.shape[2])
        X8 = torch.cat((X2_1, X7), dim=1)
        X8 = self.UpDcBlock3(X8)
        # X8 = self.dropout(X8)

        #
        # X8 = nn.functional.interpolate(X8,X1_1.shape[2])
        X8 = self.up(X8)
        X8 = self.ConvTrans4(X8)
        X1_1 = self.skipPath1(X1_1)
        X8 = nn.functional.interpolate(X8, X1_1.shape[2])
        X9 = torch.cat((X1_1, X8), dim=1)
        X9 = self.UpDcBlock4(X9)
        # X9 = self.dropout(X9)
        # X9 = self.output(X9)
        X9 = self.sig(self.output(X9))
        return X9

#
# if __name__ == "__main__":
#     x = torch.randn((1, 3, 572, 572))
#     model = UnetVariant(3, 1)
#     print(model)
#     print(model(x).shape)
