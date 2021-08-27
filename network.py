



from unet_parts import *

from GLAttention import *
from StageInfoemationTransferModule import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.attention1 = AttentionBlock(128)
        self.STAM1 = SITM(in_channel=128,hidden_channel=64)

        self.MAM2 = MAM()
        self.down2 = Down(128, 256)
        self.attention2 = AttentionBlock(256)
        self.STAM2 = SITM(256,128)

        self.MAM3 = MAM()
        self.down3 = Down(256, 512)
        self.attention3 = AttentionBlock(512)
        self.STAM3 = SITM(512, 256)

        self.MAM4 = MAM()
        self.down4 = Down(512, 512)
        self.attention4 = AttentionBlock(512)
        self.STAM4 = SITM(512, 512)
        self.MAM5 = MAM()

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.gattention = Globel_Attention(c=64,s=16384,k=90)
        self.outc = OutConv(64, n_classes)


    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.gattention(x1)
        x2 = self.down1(x1)

        x2 = self.attention1(x2)
        s2 = self.STAM1(x2,x1)

        x3 = self.MAM2(x2,s2)
        x3 = self.down2(x3)
        x3 = self.attention2(x3)
        s3 = self.STAM2(x3,s2)

        x4 = self.MAM3(x3,s3)
        x4 = self.down3(x4)
        x4 = self.attention3(x4)
        s4 = self.STAM3(x4,s3)

        x5= self.MAM4(x4,s4)
        x5 = self.down4(x5)
        x5 = self.attention4(x5)
        s5 = self.STAM4(x5,s4)

        up = self.MAM5(x5,s5)

        x = self.up1(up, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits




