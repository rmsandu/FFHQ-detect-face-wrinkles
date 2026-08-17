import numpy as np

from face_detection import calculate_wrinkle_metrics


def test_calculate_wrinkle_metrics_known_ratio():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 1  # 10 out of 100 pixels set -> 10%
    assert calculate_wrinkle_metrics(mask) == 10.0


def test_calculate_wrinkle_metrics_empty_mask_is_zero():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert calculate_wrinkle_metrics(mask) == 0.0


def test_calculate_wrinkle_metrics_full_mask_is_hundred():
    mask = np.ones((10, 10), dtype=np.uint8)
    assert calculate_wrinkle_metrics(mask) == 100.0


def test_calculate_wrinkle_metrics_squeezes_batch_and_channel_dims():
    mask_2d = np.zeros((10, 10), dtype=np.uint8)
    mask_2d[0, :] = 1
    mask_3d = mask_2d[np.newaxis, :, :]  # (1, H, W)
    mask_4d = mask_2d[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    result_2d = calculate_wrinkle_metrics(mask_2d)
    result_3d = calculate_wrinkle_metrics(mask_3d)
    result_4d = calculate_wrinkle_metrics(mask_4d)

    assert result_2d == result_3d == result_4d
