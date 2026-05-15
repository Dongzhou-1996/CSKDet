import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile



def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_places, places, stride=1, downsampling=False):
        super().__init__()
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_places, places, 1, 1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(places, places, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(places, places * self.expansion, 1, 1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_places, places * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.expansion = 4

        self.conv1 = Conv1(3, 64)
        self.layer1 = self.make_layer(64, 64, blocks[0], stride=1)
        self.layer2 = self.make_layer(256, 128, blocks[1], stride=2)
        self.layer3 = self.make_layer(512, 256, blocks[2], stride=2)
        self.layer4 = self.make_layer(1024, 512, blocks[3], stride=2)

    def make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        layers.append(Bottleneck(in_planes, planes, stride, downsampling=True))
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * self.expansion, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)      # [B, 64, 56, 56]
        x2 = self.layer1(x1)    # [B, 256, 56, 56]
        x3 = self.layer2(x2)    # [B, 512, 28, 28]
        x4 = self.layer3(x3)    # [B, 1024, 14, 14]
        x5 = self.layer4(x4)    # [B, 2048, 7, 7]
        return x1, x2, x3, x4, x5

def ResNet50():
    return ResNet([3, 4, 6, 3])


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = ResNet50()

        self.up1 = Up(2048 + 1024, 1024)
        self.up2 = Up(1024 + 512, 512)
        self.up3 = Up(512 + 256, 64)
        self.up4 = Up(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


