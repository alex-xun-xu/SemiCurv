""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from torch import nn
import torch as ch

from unet_parts import *
import numpy as np

import AutoAugmentation as AA   # import auto augmentation

# from Simplified.Network.unet_parts import DoubleConv
init_weight = {
    'glorot_uniform' : nn.init.xavier_uniform_,
    'glorot_normal' : nn.init.xavier_normal_,
    'he_normal' : nn.init.kaiming_normal_,
    'lecun_normal' : nn.init.xavier_normal_, #don't have lecun_normal in Pytorch
}

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, ki='glorot_uniform'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.ResBlock1 = ResDoubleConv(n_channels, 32)
        self.Downscale1 = Downscale(2)
        self.ResBlock2 = ResDoubleConv(32, 64)
        self.Downscale2 = Downscale(2)
        self.ResBlock3 = ResDoubleConv(64, 128)
        self.Downscale3 = Downscale(2)
        self.ResBlock4 = ResDoubleConv(128, 256)
        self.Downscale4 = Downscale(2)
        self.ResBlock5 = ResDoubleConv(256, 512)
        self.Upscale5 = Upscale(2)
        self.ResBlock6 = ResDoubleConv(512+256, 256)
        self.Upscale6 = Upscale(2)
        self.ResBlock7 = ResDoubleConv(256+128, 128)
        self.Upscale7 = Upscale(2)
        self.ResBlock8 = ResDoubleConv(128+64, 64)
        self.Upscale8 = Upscale(2)
        self.ResBlock9 = ResDoubleConv(64+32, 32)
        self.Output = OutputLayers(32, n_classes)

        self._init_weights(ki)


    def _init_weights(self, ki):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight[ki](m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ## Encoder Net
        out = self.ResBlock1(x)
        res1 = out.clone()
        out = self.Downscale1(out)

        out = self.ResBlock2(out)
        res2 = out.clone()
        out = self.Downscale2(out)

        out = self.ResBlock3(out)
        res3 = out.clone()
        out = self.Downscale3(out)

        out = self.ResBlock4(out)
        res4 = out.clone()
        out = self.Downscale4(out)

        ## Decoder Net
        out = self.ResBlock5(out)
        out = self.Upscale5(out)

        out = self.ResBlock6(ch.cat([out,res4],dim=1))
        out = self.Upscale6(out)

        out = self.ResBlock7(ch.cat([out,res3],dim=1))
        out = self.Upscale7(out)

        out = self.ResBlock8(ch.cat([out,res2],dim=1))
        out = self.Upscale8(out)

        out = self.ResBlock9(ch.cat([out,res1],dim=1))

        logits = self.Output(out)
        return logits

class ResUNet_AutoAug(nn.Module):
    def __init__(self, n_channels, n_classes, aug_para, bilinear=True, ki='glorot_uniform'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.aug_para = aug_para

        self.ResBlock1 = ResDoubleConv(n_channels, 32)
        self.Downscale1 = Downscale(2)
        self.ResBlock2 = ResDoubleConv(32, 64)
        self.Downscale2 = Downscale(2)
        self.ResBlock3 = ResDoubleConv(64, 128)
        self.Downscale3 = Downscale(2)
        self.ResBlock4 = ResDoubleConv(128, 256)
        self.Downscale4 = Downscale(2)
        self.ResBlock5 = ResDoubleConv(256, 512)
        self.Upscale5 = Upscale(2)
        self.ResBlock6 = ResDoubleConv(512+256, 256)
        self.Upscale6 = Upscale(2)
        self.ResBlock7 = ResDoubleConv(256+128, 128)
        self.Upscale7 = Upscale(2)
        self.ResBlock8 = ResDoubleConv(128+64, 64)
        self.Upscale8 = Upscale(2)
        self.ResBlock9 = ResDoubleConv(64+32, 32)
        self.Output = OutputLayers(32, n_classes)

        self._init_weights(ki)

        self.Augmentation = AA.AutoAug(aug_para)


    def _init_weights(self, ki):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight[ki](m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        ## Apply augmentation to input image
        x, Ts = self.Augmentation(x)

        ## Encoder Net
        out = self.ResBlock1(x)
        res1 = out.clone()
        out = self.Downscale1(out)

        out = self.ResBlock2(out)
        res2 = out.clone()
        out = self.Downscale2(out)

        out = self.ResBlock3(out)
        res3 = out.clone()
        out = self.Downscale3(out)

        out = self.ResBlock4(out)
        res4 = out.clone()
        out = self.Downscale4(out)

        ## Decoder Net
        out = self.ResBlock5(out)
        out = self.Upscale5(out)

        out = self.ResBlock6(ch.cat([out,res4],dim=1))
        out = self.Upscale6(out)

        out = self.ResBlock7(ch.cat([out,res3],dim=1))
        out = self.Upscale7(out)

        out = self.ResBlock8(ch.cat([out,res2],dim=1))
        out = self.Upscale8(out)

        out = self.ResBlock9(ch.cat([out,res1],dim=1))

        logits = self.Output(out)
        return logits

class ResUNet_Location(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, ki='glorot_uniform'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.ResBlock1 = ResDoubleConv(n_channels, 32)
        self.Downscale1 = Downscale(2)
        self.ResBlock2 = ResDoubleConv(32+2, 64)
        self.Downscale2 = Downscale(2)
        self.ResBlock3 = ResDoubleConv(64+2, 128)
        self.Downscale3 = Downscale(2)
        self.ResBlock4 = ResDoubleConv(128+2, 256)
        self.Downscale4 = Downscale(2)
        self.ResBlock5 = ResDoubleConv(256, 512)
        self.Upscale5 = Upscale(2)
        self.ResBlock6 = ResDoubleConv(512+256, 256)
        self.Upscale6 = Upscale(2)
        self.ResBlock7 = ResDoubleConv(256+128, 128)
        self.Upscale7 = Upscale(2)
        self.ResBlock8 = ResDoubleConv(128+64, 64)
        self.Upscale8 = Upscale(2)
        self.ResBlock9 = ResDoubleConv(64+32, 32)
        self.Output = OutputLayers(32, n_classes)

        self._init_weights(ki)


    def _init_weights(self, ki):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight[ki](m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def AppendLocationMap(self,Input):
        '''
        Generate location map
        :param x: Input tensor
        :return:
        '''

        x, y = np.meshgrid(np.arange(0, Input.shape[3]),
                           np.arange(0, Input.shape[2]))
        x = x/x.max()
        y = y/y.max()
        x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
        y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
        x = ch.tensor(np.tile(x, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        y = ch.tensor(np.tile(y, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        return ch.cat([Input, x, y], dim=1)


    def forward(self, x):
        ## Encoder Net
        out = self.ResBlock1(x)
        res1 = out.clone()
        out = self.Downscale1(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock2(out)
        res2 = out.clone()
        out = self.Downscale2(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock3(out)
        res3 = out.clone()
        out = self.Downscale3(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock4(out)
        res4 = out.clone()
        out = self.Downscale4(out)

        ## Decoder Net
        out = self.ResBlock5(out)
        out = self.Upscale5(out)

        out = self.ResBlock6(ch.cat([out,res4],dim=1))
        out = self.Upscale6(out)

        out = self.ResBlock7(ch.cat([out,res3],dim=1))
        out = self.Upscale7(out)

        out = self.ResBlock8(ch.cat([out,res2],dim=1))
        out = self.Upscale8(out)

        out = self.ResBlock9(ch.cat([out,res1],dim=1))

        logits = self.Output(out)
        return logits


class ResUNet_SinusoidLocation(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, ki='glorot_uniform',SinPeriod=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.SinPeriod = SinPeriod

        self.ResBlock1 = ResDoubleConv(n_channels, 32)
        self.Downscale1 = Downscale(2)
        self.ResBlock2 = ResDoubleConv(32+2, 64)
        self.Downscale2 = Downscale(2)
        self.ResBlock3 = ResDoubleConv(64+2, 128)
        self.Downscale3 = Downscale(2)
        self.ResBlock4 = ResDoubleConv(128+2, 256)
        self.Downscale4 = Downscale(2)
        self.ResBlock5 = ResDoubleConv(256, 512)
        self.Upscale5 = Upscale(2)
        self.ResBlock6 = ResDoubleConv(512+256, 256)
        self.Upscale6 = Upscale(2)
        self.ResBlock7 = ResDoubleConv(256+128, 128)
        self.Upscale7 = Upscale(2)
        self.ResBlock8 = ResDoubleConv(128+64, 64)
        self.Upscale8 = Upscale(2)
        self.ResBlock9 = ResDoubleConv(64+32, 32)
        self.Output = OutputLayers(32, n_classes)

        self._init_weights(ki)


    def _init_weights(self, ki):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight[ki](m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def AppendLocationMap(self,Input):
        '''
        Generate location map
        :param x: Input tensor
        :return:
        '''

        x, y = np.meshgrid(np.arange(0, Input.shape[3]),
                           np.arange(0, Input.shape[2]))
        x = np.sin(x/x.max()*self.SinPeriod*np.pi)
        y = np.cos(y/y.max()*self.SinPeriod*np.pi)
        x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
        y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
        x = ch.tensor(np.tile(x, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        y = ch.tensor(np.tile(y, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        return ch.cat([Input, x, y], dim=1)


    def forward(self, x):
        ## Encoder Net
        out = self.AppendLocationMap(x)
        out = self.ResBlock1(out)
        res1 = out.clone()
        out = self.Downscale1(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock2(out)
        res2 = out.clone()
        out = self.Downscale2(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock3(out)
        res3 = out.clone()
        out = self.Downscale3(out)

        out = self.AppendLocationMap(out)
        out = self.ResBlock4(out)
        res4 = out.clone()
        out = self.Downscale4(out)

        ## Decoder Net
        out = self.ResBlock5(out)
        out = self.Upscale5(out)

        out = self.ResBlock6(ch.cat([out,res4],dim=1))
        out = self.Upscale6(out)

        out = self.ResBlock7(ch.cat([out,res3],dim=1))
        out = self.Upscale7(out)

        out = self.ResBlock8(ch.cat([out,res2],dim=1))
        out = self.Upscale8(out)

        out = self.ResBlock9(ch.cat([out,res1],dim=1))

        logits = self.Output(out)
        return logits

class HED(nn.Module):
    def __init__(self, n_channels=3):
        super(HED, self).__init__()

        self.n_channels = n_channels

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            # torch.nn.Sigmoid()
        )

        # self.load_state_dict({ strKey.replace('module', 'net'):
        #                            tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(
        #     url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch',
        #     file_name='hed-' + arguments_strModel).items() })
        self._init_weights('glorot_uniform')


    def _init_weights(self, ki):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight[ki](m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tenInput):
        # tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        # tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        # tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434
        #
        # tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))


class RCF(nn.Module):
    def __init__(self, n_channels):
        super(RCF, self).__init__()

        self.n_channels = n_channels

        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(self.n_channels, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results

import sys
sys.path.append('/vision02/SSL/Simplified/CompetingMethods/pytorch-deeplab-xception')
from modeling.deeplab import *
class DeepLabV3Plus_Exception(DeepLab):
    def __init__(self, n_channels, n_classes = 1):
        super(DeepLabV3Plus_Exception, self).__init__(num_classes=n_classes,backbone='xception')

        self.n_channels = n_channels




def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


def crop_caffe(location, variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(location)
    y1 = int(location)
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(in_channels, out_channels, h, w):
    weights = np.zeros([in_channels, out_channels, h, w])
    if in_channels != out_channels:
        raise ValueError("Input Output channel!")
    if h != w:
        raise ValueError("filters need to be square!")
    filt = upsample_filt(h)
    weights[range(in_channels), range(out_channels), :, :] = filt
    return np.float32(weights)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)

class UNet_bk(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


from torchvision import models
import math


class LinkNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)

        if in_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(in_channels, filters[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.return_features = False
        self.tanh = nn.Tanh()

        for m in [self.finaldeconv1, self.finalconv2]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = (
                self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
                ]
                + e3
        )
        d3 = (
                self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
                ]
                + e2
        )
        d2 = (
                self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
                ]
                + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5[:, :, :rows, :cols]

class LinkNet34_SinusoidLocation(nn.Module):
    def __init__(self, in_channels=3+2, num_classes=2, SinPeriod=4):
        super(LinkNet34_SinusoidLocation, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.SinPeriod = SinPeriod

        if in_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(in_channels, filters[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = self.EncLayer1()
        self.encoder2 = self.EncLayer2()
        self.encoder3 = self.EncLayer3()
        self.encoder4 = self.EncLayer4()

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.return_features = False
        self.tanh = nn.Tanh()

        for m in [self.finaldeconv1, self.finalconv2]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def BasicBlock(self,in_dim,out_dim, kernel_size=3,stride1=2,stride2=1):
    #
    #     return torch.nn.sequential(
    #     torch.nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, kernel_size), stride=(stride1, stride1), padding=(1, 1), bias=False),
    #     torch.nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     torch.nn.ReLU(inplace=True),
    #     torch.nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size, kernel_size), stride=(stride2, stride2), padding=(1, 1), bias=False),
    #     torch.nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))



    def conv1x1(self, in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def EncLayer1(self):

        downsample = nn.Sequential(
            self.conv1x1(64+2, 64, 1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        return torch.nn.Sequential(
            # Block 1
            BasicBlock(inplanes=64+2, planes=64, stride=1, downsample=downsample),
            # Block 2
            BasicBlock(inplanes=64, planes=64, stride=1, downsample=None),
            # Block 3
            BasicBlock(inplanes=64, planes=64, stride=1, downsample=None),
        )

    def EncLayer2(self):

        downsample = nn.Sequential(
            self.conv1x1(64+2, 128, 2),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        return torch.nn.Sequential(
            # BasicBlock
            BasicBlock(inplanes=64+2, planes=128, stride=2, downsample=downsample),

            # BasicBlock(
            BasicBlock(inplanes=128, planes=128, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=128, planes=128, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=128, planes=128, stride=1, downsample=None)

            )

    def EncLayer3(self):

        downsample = nn.Sequential(
            self.conv1x1(128+2, 256, 2),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        return torch.nn.Sequential(
            # BasicBlock
            BasicBlock(inplanes=128+2, planes=256, stride=2, downsample=downsample),

            # BasicBlock(
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None),

            )

    def EncLayer4(self):

        downsample = nn.Sequential(
            self.conv1x1(256+2, 512, 2),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        return torch.nn.Sequential(
            # BasicBlock
            BasicBlock(inplanes=256+2, planes=512, stride=2, downsample=downsample),

            # BasicBlock(
            BasicBlock(inplanes=512, planes=512, stride=1, downsample=None),

            # BasicBlock(
            BasicBlock(inplanes=512, planes=512, stride=1, downsample=None),

            )

    def AppendLocationMap(self,Input):
        '''
        Generate location map
        :param x: Input tensor
        :return:
        '''

        x, y = np.meshgrid(np.arange(0, Input.shape[3]),
                           np.arange(0, Input.shape[2]))
        x = np.sin(x/x.max()*self.SinPeriod*np.pi)
        y = np.cos(y/y.max()*self.SinPeriod*np.pi)
        x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
        y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
        x = ch.tensor(np.tile(x, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        y = ch.tensor(np.tile(y, [Input.shape[0], 1, 1, 1]),
                         device=Input.device)
        return ch.cat([Input, x, y], dim=1)

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.AppendLocationMap(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        x = self.AppendLocationMap(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.AppendLocationMap(e1))
        e3 = self.encoder3(self.AppendLocationMap(e2))
        e4 = self.encoder4(self.AppendLocationMap(e3))

        # Decoder with Skip Connections
        d4 = (
                self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
                ]
                + e3
        )
        d3 = (
                self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
                ]
                + e2
        )
        d2 = (
                self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
                ]
                + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5[:, :, :rows, :cols]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def conv3x3(self,in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(self,in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LinkNet34MTL(nn.Module):
    def __init__(self, task1_classes=2, task2_classes=37):
        super(LinkNet34MTL, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Decoder
        self.a_decoder4 = DecoderBlock(filters[3], filters[2])
        self.a_decoder3 = DecoderBlock(filters[2], filters[1])
        self.a_decoder2 = DecoderBlock(filters[1], filters[0])
        self.a_decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.a_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.a_finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv2 = nn.Conv2d(32, 32, 3)
        self.a_finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        for m in [
            self.finaldeconv1,
            self.finalconv2,
            self.a_finaldeconv1,
            self.a_finalconv2,
        ]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = (
            self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        d3 = (
            self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        d2 = (
            self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        # Decoder with Skip Connections
        a_d4 = (
            self.a_decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        a_d3 = (
            self.a_decoder3(a_d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        a_d2 = (
            self.a_decoder2(a_d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        a_d1 = self.a_decoder1(a_d2)

        # Final Classification
        a_f1 = self.a_finaldeconv1(a_d1)
        a_f2 = self.a_finalrelu1(a_f1)
        a_f3 = self.a_finalconv2(a_f2)
        a_f4 = self.a_finalrelu2(a_f3)
        a_f5 = self.a_finalconv3(a_f4)

        return f5[:, :, :rows, :cols], a_f5[:, :, :rows, :cols]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


