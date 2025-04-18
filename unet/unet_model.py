"""Full assembly of the parts to form the complete network"""

import torch
import torch.nn as nn
from .unet_parts import *
from torchvision.models import resnet50, ResNet50_Weights


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        pretrained=True,
        freeze_encoder=False,
        use_attention=False,
    ):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained ResNet50 as encoder
        if pretrained is True:
            # Load pretrained weights
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            # Randomly initialize weights
            resnet = resnet50(weights=None)
            print("Randomly initialized weights")

        # Initial layers from ResNet50
        self.input_layer = nn.Sequential(
            resnet.conv1,  # 3 -> 64
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool  # 64 -> 64, size halves
        # Encoder from ResNet50
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        if self.use_attention:
            print("Using attention gates")
            # Attention mechanism**
            self.att1 = AttentionGate(
                F_g=2048, F_l=1024, F_int=512
            )  # Between encoder4 and encoder3
            self.att2 = AttentionGate(
                F_g=1024, F_l=512, F_int=256
            )  # Between up1 output and encoder2
            self.att3 = AttentionGate(
                F_g=512, F_l=256, F_int=128
            )  # Between up2 output and encoder1
            self.att4 = AttentionGate(
                F_g=256, F_l=64, F_int=32
            )  # Between up3 output and encoder0
        else:
            print("Not using attention gates")
            # When attention is disabled, simply pass the features through
            self.att1 = self.att2 = self.att3 = self.att4 = nn.Identity()

        # Freeze encoder if desired
        if freeze_encoder is True:
            # Freeze all layers
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            for param in resnet.parameters():
                param.requires_grad = True

        # Initialize weights for custom layers if not pretrained
        if not pretrained:
            self._initialize_weights()
        # Decoder path

        self.up1 = Up(
            x1_channels=2048, x2_channels=1024, out_channels=1024, bilinear=bilinear
        )
        self.up2 = Up(1024, 512, 512, bilinear=bilinear)
        self.up3 = Up(512, 256, 256, bilinear=bilinear)
        self.up4 = Up(256, 64, 128, bilinear=bilinear)
        self.up5 = Up(128, 0, 64, bilinear=bilinear)

        # Final output layer

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path

        # Initial layers
        x0 = self.input_layer(x)  # 64x256x256
        x1 = self.encoder1(self.pool(x0))  # 256x128x128
        x2 = self.encoder2(x1)  # 512x64x64
        x3 = self.encoder3(x2)  # 1024x32x32
        x4 = self.encoder4(x3)  # 2048x16x16
        # print out the shapes of each encoder layer
        print(
            f"Encoder shapes: x0: {x0.shape}, x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}"
        )

        # Apply attention gates to skip connections
        x3_att = self.att1(g=x4, x=x3)  # Attention between encoder4 and encoder3
        x2_att = self.att2(g=x3, x=x2)  # Attention between encoder3 and encoder2
        x1_att = self.att3(g=x2, x=x1)  # Attention between encoder2 and encoder1
        x0_att = self.att4(g=x1, x=x0)  # Attention between encoder1 and encoder0

        # Decoder with attention-gated skip connections

        x = self.up1(x4, x3_att)
        x = self.up2(x, x2_att)
        x = self.up3(x, x1_att)
        x = self.up4(x, x0_att)
        x = self.up5(x, None)
        print(f"Decoder shape after up5: {x.shape}")
        logits = self.outc(x)
        return (
            logits if self.n_classes == 1 else torch.softmax(logits, dim=1)
        )  # -> 1x512x512

    def _initialize_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
