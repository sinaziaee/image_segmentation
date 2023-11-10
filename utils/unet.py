import torch.nn as nn
from torchvision import transforms
import torch

def double_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(p=0.2),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1):
        super(UNet, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting Path
        # Each convolution is applied twice
        self.down_conv1 = double_convolution(input_channels, 32)
        self.down_conv2 = double_convolution(32, 64)
        self.down_conv3 = double_convolution(64, 128)
        self.down_conv4 = double_convolution(128, 256)
        self.down_conv5 = double_convolution(256, 512)
        # Expanding Path
        self.up_transpose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # Below, in_channels become 1024 as we are concatinating
        self.up_conv1 = double_convolution(512, 256)
        self.up_transpose2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv2 = double_convolution(256, 128)
        self.up_transpose3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv3 = double_convolution(128, 64)
        self.up_transpose4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up_conv4 = double_convolution(64, 32)
        # out_conv
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        down1 = self.down_conv1(x)
        down2 = self.maxpool2d(down1)
        down3 = self.down_conv2(down2)
        down4 = self.maxpool2d(down3)
        down5 = self.down_conv3(down4)
        down6 = self.maxpool2d(down5)
        down7 = self.down_conv4(down6)
        down8 = self.maxpool2d(down7)
        down9 = self.down_conv5(down8)
        # decoder
        up1 = self.up_transpose1(down9)
        x = self.up_conv1(torch.cat([down7, up1], dim=1))
        up2 = self.up_transpose2(x)
        x = self.up_conv2(torch.cat([down5, up2], dim=1))
        up3 = self.up_transpose3(down5)
        x = self.up_conv3(torch.cat([down3, up3], dim=1))
        up4 = self.up_transpose4(down3)
        x = self.up_conv4(torch.cat([down1, up4], dim=1))

        out = self.conv_out(x)
        return out