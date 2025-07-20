# Fixed cbam.py with correct import

import torch
import torch.nn as nn

from models.basic_conv import BasicConv  # Fixed import


class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)


class SpatialGate(nn.Module):
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