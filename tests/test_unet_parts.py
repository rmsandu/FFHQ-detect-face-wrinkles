import torch

from unet.unet_parts import AttentionGate, DoubleConv, Down, OutConv, Up


def test_double_conv_output_shape_and_channels():
    layer = DoubleConv(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32, 32)


def test_down_halves_spatial_dims():
    layer = Down(in_channels=8, out_channels=16)
    x = torch.randn(2, 8, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 16, 16)


def test_up_transposed_conv_with_skip_connection():
    layer = Up(x1_channels=32, x2_channels=16, out_channels=16, bilinear=False)
    x1 = torch.randn(2, 32, 8, 8)
    x2 = torch.randn(2, 16, 16, 16)
    out = layer(x1, x2)
    assert out.shape == (2, 16, 16, 16)


def test_up_bilinear_with_skip_connection():
    layer = Up(x1_channels=32, x2_channels=16, out_channels=16, bilinear=True)
    x1 = torch.randn(2, 32, 8, 8)
    x2 = torch.randn(2, 16, 16, 16)
    out = layer(x1, x2)
    assert out.shape == (2, 16, 16, 16)


def test_up_without_skip_connection():
    layer = Up(x1_channels=32, x2_channels=0, out_channels=16, bilinear=False)
    x1 = torch.randn(2, 32, 8, 8)
    out = layer(x1, None)
    assert out.shape == (2, 16, 16, 16)


def test_out_conv_output_channels():
    layer = OutConv(in_channels=16, out_channels=1)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 1, 32, 32)


def test_attention_gate_preserves_skip_shape():
    gate = AttentionGate(F_g=32, F_l=16, F_int=8)
    g = torch.randn(2, 32, 8, 8)
    x = torch.randn(2, 16, 16, 16)
    out = gate(g=g, x=x)
    assert out.shape == x.shape
