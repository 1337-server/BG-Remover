# ruff: noqa: F821

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
#  Basic building blocks (RSU modules)
# ---------------------------------------------------------
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.size()[2:], mode='bilinear', align_corners=False)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.size()[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.size()[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.size()[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.size()[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

# Shorter RSUs omitted for brevity (U²-Net uses multiple depths)
# ---------------------------------------------------------
#  U2NET full architecture
# ---------------------------------------------------------
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU7(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU7(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU7(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU7(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU7(512, 256, 512)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # upsample and side outputs
        d1 = self.side1(hx1)
        d2 = F.interpolate(self.side2(hx2), scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(hx3), scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.side4(hx4), scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.side5(hx5), scale_factor=16, mode='bilinear', align_corners=False)
        d6 = F.interpolate(self.side6(hx6), scale_factor=32, mode='bilinear', align_corners=False)

        d0 = torch.sigmoid((d1 + d2 + d3 + d4 + d5 + d6) / 6)
        return d0, d1, d2, d3, d4, d5, d6



# ---------- Lightweight U^2-Net (U2NETP) ----------

class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU7(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU7(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU7(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4F(128, 16, 64)
        self.stage3d = RSU4F(128, 16, 64)
        self.stage2d = RSU4F(128, 16, 64)
        self.stage1d = RSU4F(128, 16, 64)

        # side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)

        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))

        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d1 = _upsample_like(d1, x)
        d2 = _upsample_like(d2, x)
        d3 = _upsample_like(d3, x)
        d4 = _upsample_like(d4, x)
        d5 = _upsample_like(d5, x)
        d6 = _upsample_like(d6, x)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
