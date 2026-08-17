import torch

from unet import UNet


def _make_model(**kwargs):
    defaults = dict(
        n_channels=3,
        n_classes=1,
        bilinear=False,
        pretrained=False,  # avoid downloading torchvision weights in CI
    )
    defaults.update(kwargs)
    return UNet(**defaults)


def test_forward_pass_output_shape():
    model = _make_model()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_forward_pass_with_attention_enabled():
    model = _make_model(use_attention=True)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_freeze_encoder_disables_grad():
    model = _make_model(freeze_encoder=True)
    encoder_params = list(model.encoder1.parameters()) + list(
        model.encoder4.parameters()
    )
    assert all(not p.requires_grad for p in encoder_params)


def test_unfrozen_encoder_keeps_grad():
    model = _make_model(freeze_encoder=False)
    encoder_params = list(model.encoder1.parameters())
    assert all(p.requires_grad for p in encoder_params)
