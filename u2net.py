# u2net.py
# Minimal implementation of U2NET and U2NETP (for background removal)

import torch
import torch.nn as nn

# Basic building block
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Basic U2NET block
class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
    def forward(self, x):
        return self.rebnconvin(x)

# Full-size network
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
    def forward(self, x):
        x1 = self.stage1(x)
        d1 = self.side1(x1)
        return d1

# Lightweight version
class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()
        self.stage1 = RSU7(in_ch, 16, 32)
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
    def forward(self, x):
        x1 = self.stage1(x)
        d1 = self.side1(x1)
        return d1
