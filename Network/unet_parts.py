""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

#### Operations defined for ResUnet
class ResDoubleConv(nn.Module):
    ''' Residual Double Convolution Block '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99),
        )
        self.conv_bn_2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
        )
        # residual connection within resblock
        self.identity_bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)  # eps=1e-05, momentum=0.1
        self.identity = nn.Conv2d(in_channels, out_channels, 1)  # indentity mapping
        # activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # res = nn.Identity(x)
        out = self.activation(self.conv_bn_1(x))
        out = self.activation(self.conv_bn_2(out)+self.identity_bn(self.identity(x)))
        return out

class ResSingleConv(nn.Module):
    ''' Residual Double Convolution Block '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99),
        )
        self.conv_bn_2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # res = nn.Identity(x)
        out = self.activation(self.conv_bn_1(x))
        out = self.activation(self.conv_bn_2(out))
        return out

class Downscale(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self, factor=2):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(factor),
        )

    def forward(self, x):
        return self.maxpool(x)

class Upscale(nn.Module):
    ''' Upscaling with bilinear interpolation '''

    def __init__(self, factor=2):
        super().__init__()
        self.interpolate = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        )

    def forward(self,x):
        return self.interpolate(x)

class OutputLayers(nn.Module):
    ''' Output layers including classifier '''
    def __init__(self, in_channels, out_channels, mid_channels=3):
        super().__init__()
        self.output = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.output(x)

#### Operations defined for original Unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
