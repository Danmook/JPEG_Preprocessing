# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------本文件弃用----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch

class Smooth(nn.Module):
    def __init__(self):
        super(Smooth,self).__init__()

        # input size: 256*256
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),  # feature size: 256*256
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, stride=2, groups=3),  # feature size: 32*32
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2), # feature size: 16*16
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2), # feature size: 8*8
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),  # 16*16
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4),   # 64*64
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=4),    # 256*256
        )

    def forward(self, x):
        return self.layer(x)

