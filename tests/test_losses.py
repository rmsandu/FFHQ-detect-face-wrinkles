import pytest
import torch

from losses import binary_focal_loss, dice_loss


def test_dice_loss_perfect_overlap_is_near_zero():
    pred = torch.ones(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() < 1e-4


def test_dice_loss_no_overlap_is_near_one():
    pred = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() > 0.99


def test_dice_loss_accepts_3d_input():
    pred = torch.ones(2, 8, 8)
    target = torch.ones(2, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.dim() == 0


def test_binary_focal_loss_lower_for_correct_confident_predictions():
    target = torch.ones(4, 1, 8, 8)
    confident_correct_logits = torch.full((4, 1, 8, 8), 10.0)
    confident_wrong_logits = torch.full((4, 1, 8, 8), -10.0)

    correct_loss = binary_focal_loss(confident_correct_logits, target)
    wrong_loss = binary_focal_loss(confident_wrong_logits, target)

    assert correct_loss.item() < wrong_loss.item()


def test_binary_focal_loss_accepts_3d_input():
    logits = torch.zeros(4, 8, 8)
    target = torch.zeros(4, 8, 8)
    loss = binary_focal_loss(logits, target)
    assert loss.dim() == 0


def test_binary_focal_loss_raises_on_shape_mismatch():
    logits = torch.zeros(4, 1, 8, 8)
    target = torch.zeros(4, 1, 4, 4)
    with pytest.raises(AssertionError):
        binary_focal_loss(logits, target)
