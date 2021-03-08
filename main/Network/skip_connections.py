from abc import ABC
import torch
import torch.nn as nn


class SkipConnection01(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(SkipConnection01, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_3_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output_1 = self.conv_3_3_3(x)
        output_2 = self.conv_3_3_2(x)
        output_3 = self.conv_3_3_1(x)
        output_1 = torch.cat((output_1, output_2), dim=1)

        return torch.cat((output_1, output_3), dim=1)


class SkipConnection02(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(SkipConnection02, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_3_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output_1 = self.conv_3_3_2(x)
        output_2 = self.conv_3_3_1(x)
        output_1 = torch.cat((output_1, output_2), dim=1)

        return output_1


class SkipConnection03(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(SkipConnection03, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_3_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output_1 = self.conv_3_3_2(x)
        output_2 = self.conv_3_3_1(x)
        output_1 = torch.cat((output_1, output_2), dim=1)

        return output_1


class SkipConnection04(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(SkipConnection04, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_3_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=self.output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output_1 = self.conv_3_3_1(x)

        return output_1
