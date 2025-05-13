import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    """
    A flexible 2D convolutional block with BatchNorm and optional ReLU activation.
    """
    def __init__(self, in_c, out_c, kernel_size=3, padding=1,
                 dilation=1, groups=1, bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x

class squeeze_excitation_block(nn.Module):
    """
    Squeeze-and-Excitation block to recalibrate channel-wise feature responses.
    """
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context aggregation.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Image-level pooling branch
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0),
        )
        # Atrous conv branches
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        # 1x1 conv to merge
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

class conv_block(nn.Module):
    """
    Basic conv block: two Conv2D layers followed by an SE block.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.se = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return self.se(x)

# -------------- CBAM Spatial Gate Components --------------

class ChannelPool(nn.Module):
    """
    Pooling across channels: max and average.
    """
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)

class BasicConv(nn.Module):
    """
    Simple Conv-BN-ReLU block for SpatialGate.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    """
    Spatial attention: learns a per-pixel weight map.
    """
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            in_ch=2, out_ch=1,
            kernel_size=7, stride=1,
            padding=(7-1)//2,
            relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
