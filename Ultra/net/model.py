""" Full assembly of the parts to form the complete network """
import torch

from unet_part import DoubleConv,Down,Up,OutConv
import torchinfo
class UNet(torch.nn.Module):
    def __init__(self, n_channels:int, bilinear:bool=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels=n_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        factor = 2 if bilinear else 1
        self.down4 = Down(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up(in_channels=1024,out_channels= 512 // factor,bilinear= bilinear)
        self.up2 = Up(in_channels=512,out_channels= 256 // factor, bilinear=bilinear)
        self.up3 = Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear)
        self.up4 = Up(in_channels=128, out_channels=64,bilinear= bilinear)
        self.outc = OutConv(in_channels=64, out_channels=3)
    def forward(self, inp:torch.Tensor)->torch.Tensor:
        x = inp
        print(x.shape)
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)
        x = self.up1(x5, x4)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        x = self.outc(x) + inp
        print(x.shape)
        return x

torch.manual_seed(0)
torch.cuda.empty_cache()

data = torch.randn(1, 3, 128, 128).to("cuda")
model = UNet(n_channels=3).to("cuda")
torchinfo.summary(model=model,input_data=data,device="cuda")