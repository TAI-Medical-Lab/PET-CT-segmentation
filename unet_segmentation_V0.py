import torch
import torch.nn as nn
from config_segmentation import config
opx = config()
opt = opx()

class ChannelAttention(nn.Module):
    # def __init__(self, in_planes, ratio=16):
    def __init__(self, in_planes, ratio=opt.ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernelsize, pad):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernelsize, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernelsize, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv1(input)

class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()
        in_ch = int(in_ch/2)
        self.conv1 = DoubleConv(in_ch, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256, 3, 1)
        self.pool4 = nn.MaxPool2d(2)

       
        # self.conv5 = DoubleConv(256, 512, 3, 1)
        # self.pool5 = nn.MaxPool2d(2)

        self.conv1_B = DoubleConv(in_ch, 32, 3, 1)
        self.pool1_B = nn.MaxPool2d(2)
        self.conv2_B = DoubleConv(32, 64, 3, 1)
        self.pool2_B = nn.MaxPool2d(2)
        self.conv3_B = DoubleConv(64, 128, 3, 1)
        self.pool3_B = nn.MaxPool2d(2)
        self.conv4_B = DoubleConv(128, 256, 3, 1)
        self.pool4_B = nn.MaxPool2d(2)


        self.con1 = DoubleConv(512, 512, 1, 0)

        # 修改注意力的头数
        self.CAM = ChannelAttention(512, ratio=opt.ratio)
        self.PAM = SpatialAttention()


        self.conv5 = DoubleConv(512, 512, 3, 1)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256, 3, 1)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128, 3, 1)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64, 3, 1)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32,3, 1)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, ct, pet):
        c1 = self.conv1(ct)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        # c5 = self.conv5(p4)
        # p5 = self.pool5(c5)

        c1_B = self.conv1_B(pet)
        p1_B = self.pool1_B(c1_B)
        c2_B = self.conv2_B(p1_B)
        p2_B = self.pool2_B(c2_B)
        c3_B = self.conv3_B(p2_B)
        p3_B = self.pool3_B(c3_B)
        c4_B = self.conv4_B(p3_B)
        p4_B = self.pool4_B(c4_B)

        p5 = torch.cat([p4, p4_B], dim=1)

        # # 1*1卷积
        # p5_1 = self.con1(p5)
        #
        # #加入cbam
        # p5_ca = self.CAM(p5_1) * p5_1
        # p5_pa = self.PAM(p5_ca) * p5_ca

        c5 = self.conv5(p5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)

        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10
