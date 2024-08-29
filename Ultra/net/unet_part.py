import torch
import torch.nn as nn
import torch.nn.functional as F
from uvmb import UVMB
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels:int, out_channels:int, mid_channels:int=None):
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.ub = UVMB(c=in_channels, w=64,h=64)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels= mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        inputs = F.interpolate(input=x, size=[64, 64],   mode='bilinear', align_corners=True)
        outputs = self.ub(inputs)
        outputs = F.interpolate(input=outputs, size=[x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True) + x
        return self.double_conv(outputs)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels:int, out_channels:int):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels,out_channels= out_channels))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels:int, out_channels:int, bilinear:bool=True):
        super(Up,self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels,mid_channels= in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diff_Y = x2.size()[2] - x1.size()[2]
        diff_X = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_X // 2, diff_X - diff_X // 2,diff_Y // 2, diff_Y - diff_Y // 2])
        x = torch.cat(tensors=[x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv(x)
