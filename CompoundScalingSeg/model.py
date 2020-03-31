import math
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from modelsummary import summary


def unet_params(model_name):
  """Get U-Net params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    #   'unet-b0': (1.0, 1.0, 224, 0.2),
    #   'unet-b1': (1.0, 1.1, 240, 0.2),
    #   'unet-b2': (1.1, 1.2, 260, 0.3),
    #   'unet-b3': (1.2, 1.4, 300, 0.3),
    #   'unet-b4': (1.4, 1.8, 380, 0.4),
    #   'unet-b5': (1.6, 2.2, 456, 0.4)
        'unet-w0': (1.0, 1.0, 300, 0.2),
        'unet-w1': (1.3, 1.0, 300, 0.2),
        'unet-w2': (1.6, 1.0, 300, 0.2),
        'unet-w3': (1.9, 1.0, 300, 0.2),
        'unet-w4': (2.1, 1.0, 300, 0.2),
        'unet-r0': (1.0, 1.0, 300, 0.2),
        'unet-r1': (1.0, 1.0, 320, 0.2),
        'unet-r2': (1.0, 1.0, 360, 0.2),
        'unet-r3': (1.0, 1.0, 420, 0.2),
        'unet-r4': (1.0, 1.0, 500, 0.2),
        'unet-v1': (1.3, 1.0, 320, 0.2),
  }
  return params_dict[model_name]

def shape_matcher(layer1, layer2, dim):
    if dim == 2:
        if len(layer1.size()) == 4 and len(layer2.size()) == 4:
            if layer1.size()[1] > layer2.size()[1]:
                layer1 = layer1[:, :layer2.size()[1], :, :]
            elif layer1.size()[1] < layer2.size()[1]:
                layer2 = layer2[:, :layer1.size()[1], :, :]
            
            if layer1.size()[3] > layer2.size()[3]:
                layer1 = layer1[:, :, :, :layer2.size()[3]]
            elif layer1.size()[3] < layer2.size()[3]:
                layer2 = layer2[:, :, :, :layer1.size()[3]]

        if len(layer1.size()) == 3 and len(layer2.size()) == 3:
            if layer1.size()[0] > layer2.size()[0]:
                layer1 = layer1[:layer2.size()[0], :, :]
            elif layer1.size()[0] < layer2.size()[0]:
                layer2 = layer2[:layer1.size()[0], :, :]

            if layer1.size()[2] > layer2.size()[2]:
                layer1 = layer1[:, :, :layer2.size()[2]]
            elif layer1.size()[2] < layer2.size()[2]:
                layer2 = layer2[:, :, :layer1.size()[2]]
    elif dim == 1:
        if len(layer1.size()) == 4 and len(layer2.size()) == 4:
            if layer1.size()[2] > layer2.size()[2]:
                layer1 = layer1[:, :, :layer2.size()[2], :]
            elif layer1.size()[2] < layer2.size()[2]:
                layer2 = layer2[:, :, :layer1.size()[2], :]

            if layer1.size()[3] > layer2.size()[3]:
                layer1 = layer1[:, :, :, :layer2.size()[3]]
            elif layer1.size()[3] < layer2.size()[3]:
                layer2 = layer2[:, :, :, :layer1.size()[3]]

        elif len(layer1.size()) == 3 and len(layer2.size()) == 3:
            if layer1.size()[0] > layer2.size()[0]:
                layer1 = layer1[:, :layer2.size()[0], :]
            elif layer1.size()[0] < layer2.size()[0]:
                layer2 = layer2[:, :layer1.size()[0], :]
            
            if layer1.size()[1] > layer2.size()[1]:
                layer1 = layer1[:, :layer2.size()[1]]
            elif layer1.size()[1] < layer2.size()[1]:
                layer2 = layer2[:, :layer1.size()[1]]

    return layer1, layer2

def input_checker(layer, targ_channel):
    if len(layer.size()) == 4:
        if layer.size()[1] > targ_channel:
            layer = layer[:, :targ_channel, :, :]
        elif layer.size()[1] < targ_channel:
            cha = targ_channel - layer.size()[1]
            stack = torch.zeros(layer.size()[0], cha, layer.size()[2], layer.size()[3]).cuda()
            layer = torch.cat((layer, stack), dim=1)
    if len(layer.size()) == 3:
        if layer.size()[0] > targ_channel:
            layer = layer[:, :targ_channel, :, :]
        elif layer.size()[0] < targ_channel:
            cha = targ_channel - layer.size()[0]
            stack = torch.zeros(cha, layer.size()[1], layer.size()[2]).cuda()
            layer = torch.cat((layer, stack), dim=0)

    return layer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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
    def __init__(self, in_channels, out_channels, coef_width, bilinear=True):
        super().__init__()
        self.coef_width = coef_width
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

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
        x = input_checker(x, self.coef_width)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True,  model_param='unet-w0'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        coef_tuple = unet_params(model_param)
        self.coef_width, self.coef_depth = coef_tuple[0], coef_tuple[1]

        self.inc = DoubleConv(n_channels, round(64*self.coef_width))
        self.down1 = Down(round(64*self.coef_width), round(128*self.coef_width))
        self.down2 = Down(round(128*self.coef_width), round(256*self.coef_width))
        self.down3 = Down(round(256*self.coef_width), round(512*self.coef_width))
        self.down4 = Down(round(512*self.coef_width), round(512*self.coef_width))
        self.up1 = Up(round(1024*self.coef_width), round(256*self.coef_width), round(1024*self.coef_width), bilinear)
        self.up2 = Up(round(512*self.coef_width), round(128*self.coef_width), round(512*self.coef_width), bilinear)
        self.up3 = Up(round(256*self.coef_width), round(64*self.coef_width), round(256*self.coef_width), bilinear)
        self.up4 = Up(round(128*self.coef_width), round(64*self.coef_width), round(128*self.coef_width), bilinear)
        self.outc = OutConv(round(64*self.coef_width), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x5 = input_checker(x5, round(1024*self.coef_width))
        x = self.up1(x5, x4)
        # x = input_checker(x, round(512*self.coef_width))
        x = self.up2(x, x3)
        # x = input_checker(x, round(256*self.coef_width))
        x = self.up3(x, x2)
        # x = input_checker(x, round(128*self.coef_width))
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet()
    param = unet_params('unet-w0')

    summary(model, torch.zeros((1, 3, param[-2], param[-2])), show_input=True)
    summary(model, torch.zeros((1, 3, param[-2], param[-2])), show_input=False)
