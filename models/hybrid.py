import torch
import torch.nn as nn
from .encoder_decoder import Encoder1, Decoder1, Encoder2, Decoder2
from .blocks import ASPP, Conv2D, SpatialGate

class DoubleUNetCBAM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.e1 = Encoder1()
        self.a1 = ASPP(1024, 64)
        self.d1 = Decoder1()
        self.out1 = Conv2D(32, 1, kernel_size=1, padding=0, act=False)

        self.e2 = Encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = Decoder2()
        self.out2 = Conv2D(32, 1, kernel_size=1, padding=0, act=False)

        self.sigmoid = nn.Sigmoid()
        self.cbam    = SpatialGate()

        self.classifier = nn.Sequential(
            nn.Linear(32*224*224, 1000),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1000, 256),
            nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x5, skips1 = self.e1(x)
        x5 = self.a1(x5)
        u1 = self.d1(x5, skips1)
        m1 = self.sigmoid(self.out1(u1))
        x2_in = x * m1

        p4, skips2 = self.e2(x2_in)
        p4 = self.a2(p4)
        u2 = self.d2(p4, skips1, skips2)
        m2 = self.sigmoid(self.out2(u2))
        feat = x2_in * m2

        feat = self.cbam(feat)
        flat = feat.view(feat.size(0), -1)
        return self.classifier(flat)
