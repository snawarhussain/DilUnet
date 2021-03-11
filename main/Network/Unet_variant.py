from abc import ABC
import torch
import torch.nn as nn
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
    def __init__(self, in_c, interm_c, out_c):
        super(DcBlock, self).__init__()
        self.in_c = in_c
        self.interm_c = interm_c
        self.out_c = out_c
        self.dilated_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_c, out_channels=self.interm_c, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(num_features=self.interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.interm_c, out_channels=self.out_c, kernel_size=3,
                      stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)

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

        self.DcBlock2 = DcBlock(32, 48, 64)
        self.skipPath2 = SkipConnection02(64, 64)

        self.DcBlock3 = DcBlock(64, 96, 128)
        self.skipPath3 = SkipConnection03(128, 128)

        self.DcBlock4 = DcBlock(128, 192, 256)
        self.skipPath4 = SkipConnection04(256, 256)


        self.DcBlock5 = DcBlock(256, 384, 512)
        # r Decoder part
        self.ConvTrans1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, )
        self.UpDcBlock1 = DcBlock(512, 384, 256)

        self.ConvTrans2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, )
        self.UpDcBlock2 = DcBlock(384, 192, 128)

        self.ConvTrans3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, )
        self.UpDcBlock3 = DcBlock(192, 96, 64)

        self.ConvTrans4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, )
        self.UpDcBlock4 = DcBlock(2*64, 48, 32)

        self.output = nn.ConvTranspose2d(32, out_channels=self.output_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X1 = self.DcBlock1(X)
        X1 = self.dropout(X1)
        X2 = nn.functional.max_pool2d(X1, kernel_size=2, stride=2)
        X2 = self.DcBlock2(X2)
        X2 = self.dropout(X2)
        X3 = nn.functional.max_pool2d(X2, kernel_size=2, stride=2)
        X3 = self.DcBlock3(X3)
        X3 = self.dropout(X3)
        X4 = nn.functional.max_pool2d(X3, kernel_size=2, stride=2)
        X4 = self.DcBlock4(X4)
        X4 = self.dropout(X4)
        X5 = nn.functional.max_pool2d(X4, kernel_size=2, stride=2)
        X5 = self.DcBlock5(X5)
        X5 = self.dropout(X5)
        # Decoder part
        X5 = self.ConvTrans1(X5)
        X5 = nn.functional.interpolate(X5, X4.shape[2])
        X4 = self.skipPath4(X4)
        X6 = torch.cat((X4, X5), dim=1)
        X6 = self.UpDcBlock1(X6)
        X6 = self.dropout(X6)

        X6 = self.ConvTrans2(X6)
        X6 = nn.functional.interpolate(X6, X3.shape[2])
        X3 = self.skipPath3(X3)
        X7 = torch.cat((X3, X6), dim=1)
        X7 = self.UpDcBlock2(X7)
        X7 = self.dropout(X7)

        X7 = self.ConvTrans3(X7)
        X7 = nn.functional.interpolate(X7,X2.shape[2])
        X2 = self.skipPath2(X2)
        X8 = torch.cat((X2, X7), dim=1)
        X8 = self.UpDcBlock3(X8)
        X8 = self.dropout(X8)

        X8 = self.ConvTrans4(X8)
        X8 = nn.functional.interpolate(X8,X1.shape[2])
        X1 = self.skipPath1(X1)
        X9 = torch.cat((X1, X8), dim=1)
        X9 = self.UpDcBlock4(X9)
        X9 = self.dropout(X9)

        X9 = self.sig(self.output(X9))
        return X9


# if __name__ == "__main__":
#     x = torch.randn((1, 3, 572, 572))
#     model = UnetVariant(3, 1)
#     print(model)
#     print(model(x).shape)
