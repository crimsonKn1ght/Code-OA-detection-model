import torch.nn as nn

from models.aspp import ASPP
from models.conv2d import Conv2D
from models.cbam import SpatialGate
from models.encoder_decoder import Encoder1, Decoder1, Encoder2, Decoder2


class DoubleUNetCBAM(nn.Module):
    def __init__(self, num_classes=3, input_size=224):
        super().__init__()
        self.input_size = input_size
        
        self.e1 = Encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = Decoder1()
        self.out1 = Conv2D(32, 1, kernel_size=1, padding=0, act=False)

        self.e2 = Encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = Decoder2()
        self.out2 = Conv2D(32, 1, kernel_size=1, padding=0, act=False)

        self.sigmoid = nn.Sigmoid()
        self.cbam = SpatialGate()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(147, 512),  # Match the actual flattened size
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # First U-Net path
        x5, skips1 = self.e1(x)
        x5 = self.a1(x5)
        u1 = self.d1(x5, skips1)
        m1 = self.sigmoid(self.out1(u1))
        
        # Apply mask to input for second path
        x2_in = x * m1

        # Second U-Net path
        p4, skips2 = self.e2(x2_in)
        p4 = self.a2(p4)
        u2 = self.d2(p4, skips1, skips2)
        m2 = self.sigmoid(self.out2(u2))
        
        # Final feature extraction and classification
        feat = x2_in * m2
        feat = self.cbam(feat)
        return self.classifier(feat)