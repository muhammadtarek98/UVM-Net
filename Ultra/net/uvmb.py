import torch
import torch.nn as nn
from mamba_ssm import Mamba
import mamba_ssm
from torchinfo import summary

class UVMB(nn.Module):
    def __init__(self, c: int = 3, w: int = 256, h: int = 256):
        super(UVMB, self).__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=w * h,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.softmax = nn.Softmax()
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.shape
        x = self.convb(x) + x
        #print(x.shape)
        x = self.ln(x.reshape(b, -1, c))
        #print(x.shape)
        y = self.model1(x).permute(0, 2, 1)
        #print(y.shape)
        z = self.model3(y).permute(0, 2, 1)
        #print(x.shape)
        att = self.softmax(self.model2(x))
        #print(att.shape)
        result = att * z
        #print(result.shape)
        output = result.reshape(b, c, w, h)
        #print(output.shape)
        output = self.smooth(output)
        #print(output.shape)
        return output

torch.manual_seed(0)
torch.cuda.empty_cache()
x = torch.randn((1, 3, 64, 64)).to("cuda")
b, c, h, w = x.shape
model = UVMB(c=c, w=w, h=h).to("cuda")
summary(model=model, input_size=x.shape,device="cuda")
