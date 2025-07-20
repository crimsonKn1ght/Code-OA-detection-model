import torch
import torch.nn as nn
from torchvision.models import resnet34
import torch.nn.functional as F

from models.conv_block import conv_block


class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()

        network = resnet34(pretrained=False)
        # print(network)

        self.x1 = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.x2 = network.layer1
        self.x3 = network.layer2
        self.x4 = network.layer3
        self.x5 = network.layer4

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1, x0]  # Added x0 (original input) to skip connections

class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64+256, 256)   # 64 from ASPP + 256 from skip s1
        self.c2 = conv_block(256+128, 128)  # 256 from c1 + 128 from skip s2
        self.c3 = conv_block(128+64, 64)    # 128 from c2 + 64 from skip s3
        self.c4 = conv_block(64+3, 32)      # 64 from c3 + 3 from resized original input

    def forward(self, x, skip):
        s1, s2, s3, s4, s0 = skip  # Now includes original input s0

        x = self.up(x)  # 7x7 -> 14x14
        x = torch.cat([x, s1], axis=1)  # s1 is 14x14
        x = self.c1(x)

        x = self.up(x)  # 14x14 -> 28x28
        x = torch.cat([x, s2], axis=1)  # s2 is 28x28
        x = self.c2(x)

        x = self.up(x)  # 28x28 -> 56x56
        x = torch.cat([x, s3], axis=1)  # s3 is 56x56
        x = self.c3(x)

        # Skip s4 since s3 and s4 are both 56x56, use s3
        x = self.up(x)  # 56x56 -> 112x112
        # Interpolate s0 to match current size
        s0_resized = nn.functional.interpolate(s0, size=(112, 112), mode='bilinear', align_corners=True)
        x = torch.cat([x, s0_resized], axis=1)
        x = self.c4(x)

        # Final upsampling to 224x224
        x = self.up(x)  # 112x112 -> 224x224
        
        return x

class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)  # 224x224 -> 224x224
        p1 = self.pool(x1)  # 224x224 -> 112x112

        x2 = self.c2(p1)  # 112x112 -> 112x112
        p2 = self.pool(x2)  # 112x112 -> 56x56

        x3 = self.c3(p2)  # 56x56 -> 56x56
        p3 = self.pool(x3)  # 56x56 -> 28x28

        x4 = self.c4(p3)  # 28x28 -> 28x28
        p4 = self.pool(x4)  # 28x28 -> 14x14

        return p4, [x4, x3, x2, x1, x0]  # [28x28, 56x56, 112x112, 224x224, 224x224]

class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Updated channel calculations based on aligned features
        self.c1 = conv_block(64+256+256, 256)  # ASPP2_up + skip1[0] + skip2[0] 
        self.c2 = conv_block(256+128+128, 128) # c1_up + skip1[1] + skip2[1]
        self.c3 = conv_block(128+64+64, 64)    # c2_up + skip1[2] + skip2[2] 
        self.c4 = conv_block(64+64+32, 32)     # c3_up + skip1[3] + skip2[3]

    def forward(self, x, skip1, skip2):
        # x: [2, 64, 14, 14] from ASPP2
        # skip1: ResNet34 features [256@14x14, 128@28x28, 64@56x56, 64@56x56, 3@224x224]  
        # skip2: Custom encoder features [256@28x28, 128@56x56, 64@112x112, 32@224x224, 3@224x224]

        # Level 1: Upsample x from 14x14 to 28x28
        x = self.up(x)  # [2, 64, 28, 28]
        
        # Align skip1[0] from 14x14 to 28x28 to match x and skip2[0]
        s1_0_aligned = F.interpolate(skip1[0], size=(28, 28), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, s1_0_aligned, skip2[0]], axis=1)  # [2, 64+256+256, 28, 28]
        x = self.c1(x)  # [2, 256, 28, 28]

        # Level 2: Upsample x from 28x28 to 56x56  
        x = self.up(x)  # [2, 256, 56, 56]
        
        # skip1[1] is already 28x28, need to upsample to 56x56
        s1_1_aligned = F.interpolate(skip1[1], size=(56, 56), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, s1_1_aligned, skip2[1]], axis=1)  # [2, 256+128+128, 56, 56]
        x = self.c2(x)  # [2, 128, 56, 56]

        # Level 3: Upsample x from 56x56 to 112x112
        x = self.up(x)  # [2, 128, 112, 112]
        
        # skip1[2] is 56x56, need to upsample to 112x112
        s1_2_aligned = F.interpolate(skip1[2], size=(112, 112), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, s1_2_aligned, skip2[2]], axis=1)  # [2, 128+64+64, 112, 112]
        x = self.c3(x)  # [2, 64, 112, 112]

        # Level 4: Upsample x from 112x112 to 224x224
        x = self.up(x)  # [2, 64, 224, 224]
        
        # skip1[3] is 56x56, need to upsample to 224x224
        s1_3_aligned = F.interpolate(skip1[3], size=(224, 224), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, s1_3_aligned, skip2[3]], axis=1)  # [2, 64+64+32, 224, 224]
        x = self.c4(x)  # [2, 32, 224, 224]

        return x
