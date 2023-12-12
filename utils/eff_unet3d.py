import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout3d(p=0.2),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se
        expansion_channels = expansion_rate * in_channels
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            assert "-- MyError --: unsupported kernel size"

        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(expansion_channels),
                nn.ReLU()
            )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm3d(expansion_channels),
            nn.ReLU()
        )
        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(in_channels=expansion_channels, out_channels=se_channels, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_channels=se_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, inputs):
        x = inputs

        if self.expansion_rate != 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.se:
            x = self.se_block(x) * x

        x = self.pointwise_conv(x)

        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs

        return x


class EffUNet3D(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock3D(32, 16, kernel_size=3, stride=1, expansion_rate=1),
            MBConvBlock3D(16, 24, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock3D(24, 24, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock3D(24, 40, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock3D(40, 40, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock3D(40, 80, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock3D(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock3D(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock3D(80, 112, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock3D(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock3D(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock3D(112, 192, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock3D(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock3D(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock3D(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock3D(192, 320, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.up_block_4 = DecoderBlock3D(432, 256)

        self.up_block_3 = DecoderBlock3D(296, 128)

        self.up_block_2 = DecoderBlock3D(152, 64)

        self.up_block_1a = DecoderBlock3D(96, 32)

        self.up_block_1b = DecoderBlock3D(32, 16)

        self.head_conv = nn.Conv3d(in_channels=16, out_channels=classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.start_conv(x)

        x2 = self.down_block_2(x1)

        x3 = self.down_block_3(x2)

        x4 = self.down_block_4(x3)

        x5 = self.down_block_5(x4)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)

        x5 = self.up_block_4(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)

        x5 = self.up_block_3(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)

        x5 = self.up_block_2(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)

        x5 = self.up_block_1a(x5)
        x5 = F.interpolate(x5, scale_factor=2)

        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output
