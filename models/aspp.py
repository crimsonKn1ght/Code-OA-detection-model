import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv2d import Conv2D


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0),
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=size, mode="bilinear", align_corners=True)
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.c5(xc)