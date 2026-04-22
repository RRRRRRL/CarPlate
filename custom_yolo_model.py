import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv + BN + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """YOLOv8-style C2f block."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, bottleneck_expansion=1.0):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        # Bottlenecks maintain channel count by default (bottleneck_expansion=1.0).
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=bottleneck_expansion) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class DecoupledHead(nn.Module):
    """Decoupled cls/box prediction head for a single scale."""

    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.cls_conv = Conv(in_channels, in_channels, 3, 1)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)

        self.box_conv = Conv(in_channels, in_channels, 3, 1)
        self.box_pred = nn.Conv2d(in_channels, 4, 1)

    def forward(self, x):
        cls_out = self.cls_pred(self.cls_conv(x))
        box_out = self.box_pred(self.box_conv(x))
        return cls_out, box_out


class ImprovedCustomYOLO(nn.Module):
    """Custom YOLO architecture with C2f backbone, SPPF, top-down FPN and multi-scale decoupled heads."""

    def __init__(self, num_classes=1):
        super().__init__()

        # Backbone
        self.stem = Conv(3, 16, 3, 2)  # P1/2
        self.layer1 = nn.Sequential(Conv(16, 32, 3, 2), C2f(32, 32, n=1, shortcut=True))  # P2/4
        self.layer2 = nn.Sequential(Conv(32, 64, 3, 2), C2f(64, 64, n=2, shortcut=True))  # P3/8
        self.layer3 = nn.Sequential(Conv(64, 128, 3, 2), C2f(128, 128, n=2, shortcut=True))  # P4/16
        self.layer4 = nn.Sequential(Conv(128, 256, 3, 2), C2f(256, 256, n=1, shortcut=True))  # P5/32
        self.sppf = SPPF(256, 256, k=5)

        # Neck (top-down FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_c2f_1 = C2f(256 + 128, 128, n=1, shortcut=False)  # P5 + P4
        self.neck_c2f_2 = C2f(128 + 64, 64, n=1, shortcut=False)  # fused P4 + P3

        # Multi-scale decoupled heads
        self.head_small = DecoupledHead(64, num_classes)  # stride 8
        self.head_medium = DecoupledHead(128, num_classes)  # stride 16
        self.head_large = DecoupledHead(256, num_classes)  # stride 32

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.sppf(self.layer4(p4))

        p4_fused = self.neck_c2f_1(torch.cat([self.upsample(p5), p4], dim=1))
        p3_fused = self.neck_c2f_2(torch.cat([self.upsample(p4_fused), p3], dim=1))

        out_small = self.head_small(p3_fused)
        out_medium = self.head_medium(p4_fused)
        out_large = self.head_large(p5)

        return out_small, out_medium, out_large
