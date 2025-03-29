"""Full assembly of the parts to form the complete network"""

import torch
import torch.nn as nn
from .unet_parts import *
from torchvision.models import resnet50, ResNet50_Weights


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, pretrained=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained ResNet50 as encoder
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # Encoder from ResNet50
        self.firstconv = resnet.conv1  # 64 channels
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # 64 channels
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        # Attention gates with adjusted channels for ResNet50
        self.attention1 = AttentionGate(F_g=2048, F_l=1024, F_int=512)
        self.attention2 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.attention3 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.attention4 = AttentionGate(F_g=256, F_l=64, F_int=32)

        # Decoder with transposed convolutions (bilinear=False)
        self.up1 = Up(3072, 1024, bilinear=False)  # 2048 + 1024 input channels
        self.up2 = Up(1536, 512, bilinear=False)  # 1024 + 512 input channels
        self.up3 = Up(768, 256, bilinear=False)  # 512 + 256 input channels
        self.up4 = Up(320, 64, bilinear=False)  # 256 + 64 input channels

        self.outc = OutConv(64, n_classes)

        # Initialize weights for non-pretrained parts
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of non-pretrained parts"""
        for m in [
            self.attention1,
            self.attention2,
            self.attention3,
            self.attention4,
            self.up1,
            self.up2,
            self.up3,
            self.up4,
            self.outc,
        ]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path with ResNet50
        x1 = self.firstrelu(self.firstbn(self.firstconv(x)))  # 64 channels
        x1_pool = self.firstmaxpool(x1)
        x2 = self.encoder1(x1_pool)  # 256 channels
        x3 = self.encoder2(x2)  # 512 channels
        x4 = self.encoder3(x3)  # 1024 channels
        x5 = self.encoder4(x4)  # 2048 channels

        # Decoder path with attention gates
        x4_att = self.attention1(g=x5, x=x4)
        x = self.up1(x5, x4_att)

        x3_att = self.attention2(g=x, x=x3)
        x = self.up2(x, x3_att)

        x2_att = self.attention3(g=x, x=x2)
        x = self.up3(x, x2_att)

        x1_att = self.attention4(g=x, x=x1)
        x = self.up4(x, x1_att)

        logits = self.outc(x)

        # Return raw logits for binary segmentation
        if self.n_classes == 1:
            return logits
        return torch.softmax(logits, dim=1)

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # Apply checkpointing to each module
        modules = [
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.up1,
            self.up2,
            self.up3,
            self.up4,
        ]
        for i, module in enumerate(modules):
            setattr(
                self,
                f"module_{i}",
                torch.utils.checkpoint.checkpoint_sequential([module], 1, module),
            )
