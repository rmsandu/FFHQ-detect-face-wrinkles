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
    ):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained ResNet50 as encoder
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

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

        # Freeze encoder if desired
        if freeze_encoder is True:
            # Freeze all layers
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            for param in resnet.parameters():
                param.requires_grad = True

        # Unfreeze the last 2 layers of the encoder when freeze_encoder is True
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

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

        # Decoder
        # print out the shape after each decoder layer

        x = self.up1(x4, x3)
        print(f"Decoder shape after up1: {x.shape}")
        x = self.up2(x, x2)
        print(f"Decoder shape after up2: {x.shape}")
        x = self.up3(x, x1)
        print(f"Decoder shape after up3: {x.shape}")
        x = self.up4(x, x0)
        print(f"Decoder shape after up4: {x.shape}")
        x = self.up5(x, None)
        # print out the shape after each decoder layer
        print(f"Decoder shape after up5: {x.shape}")
        logits = self.outc(x)
        return (
            logits if self.n_classes == 1 else torch.softmax(logits, dim=1)
        )  # -> 1x512x512
