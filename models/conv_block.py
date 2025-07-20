import torch.nn as nn

from models.conv2d import Conv2D
from models.se import squeeze_excitation_block


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.se = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return self.se(x)
