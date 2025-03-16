"""Full assembly of the parts to form the complete network"""

import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Add attention gates
        self.attention1 = AttentionGate(F_g=512 // factor, F_l=512, F_int=256)
        self.attention2 = AttentionGate(F_g=256 // factor, F_l=256, F_int=128)
        self.attention3 = AttentionGate(F_g=128 // factor, F_l=128, F_int=64)
        self.attention4 = AttentionGate(F_g=64, F_l=64, F_int=32)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply attention gates on skip connections
        x4_att = self.attention1(g=x5, x=x4)
        x = self.up1(x5, x4_att)

        x3_att = self.attention2(g=x, x=x3)
        x = self.up2(x, x3_att)

        x2_att = self.attention3(g=x, x=x2)
        x = self.up3(x, x2_att)

        x1_att = self.attention4(g=x, x=x1)
        x = self.up4(x, x1_att)

        logits = self.outc(x)
        if self.n_classes == 1:
            return torch.sigmoid(logits)  # Binary segmentation
        else:
            return torch.softmax(logits, dim=1)  # Multi-class segmentation

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # Apply checkpointing to each module
        self.inc = torch.utils.checkpoint.checkpoint_sequential([self.inc], 1, self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential(
            [self.down1], 1, self.down1
        )
        self.down2 = torch.utils.checkpoint.checkpoint_sequential(
            [self.down2], 1, self.down2
        )
        self.down3 = torch.utils.checkpoint.checkpoint_sequential(
            [self.down3], 1, self.down3
        )
        self.down4 = torch.utils.checkpoint.checkpoint_sequential(
            [self.down4], 1, self.down4
        )
        self.up1 = torch.utils.checkpoint.checkpoint_sequential([self.up1], 1, self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint_sequential([self.up2], 1, self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint_sequential([self.up3], 1, self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint_sequential([self.up4], 1, self.up4)
        self.outc = torch.utils.checkpoint.checkpoint_sequential(
            [self.outc], 1, self.outc
        )
