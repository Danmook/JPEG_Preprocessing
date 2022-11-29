# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------本文件弃用----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE,self).__init__()
        # Input:3*256*256
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),    # 128*128*128
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64), # 64*64*64
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32), # 32*32*32
        )

        self.Decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 3, 3, 1, 1), 
        )
    
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x