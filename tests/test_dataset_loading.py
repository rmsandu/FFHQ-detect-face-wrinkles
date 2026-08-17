import numpy as np
import pytest
from PIL import Image

from utils.dataset_loading import (
    WrinkleDataset,
    get_augmentation_transforms,
    get_debug_transforms,
)


def _write_pair(image_dir, mask_dir, name, size=(64, 64), mask_value=255):
    image = Image.fromarray(
        (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8), mode="RGB"
    )
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[: size[1] // 4, :] = mask_value  # a stripe of "wrinkle" pixels
    mask_img = Image.fromarray(mask, mode="L")

    image.save(image_dir / f"{name}.png")
    mask_img.save(mask_dir / f"{name}.png")


@pytest.fixture
def synthetic_dataset_dirs(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for i in range(3):
        _write_pair(image_dir, mask_dir, f"{i:05d}")
    return image_dir, mask_dir


def test_dataset_output_is_resized_to_512(synthetic_dataset_dirs):
    image_dir, mask_dir = synthetic_dataset_dirs
    dataset = WrinkleDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_debug_transforms(),
        dilate_mask=False,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)


def test_dataset_mask_values_are_binary(synthetic_dataset_dirs):
    image_dir, mask_dir = synthetic_dataset_dirs
    dataset = WrinkleDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_debug_transforms(),
        dilate_mask=True,
    )
    sample = dataset[0]
    unique_values = set(sample["mask"].unique().tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_dataset_raises_on_mismatched_pair_counts(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    _write_pair(image_dir, mask_dir, "00000")
    # Add an extra image with no corresponding mask.
    Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype(np.uint8), mode="RGB"
    ).save(image_dir / "00001.png")

    with pytest.raises(RuntimeError):
        WrinkleDataset(image_dir=image_dir, mask_dir=mask_dir)


@pytest.mark.parametrize(
    "transform_factory", [get_debug_transforms, get_augmentation_transforms]
)
def test_transform_pipelines_run_without_error(transform_factory):
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:16, :] = 1

    transformed = transform_factory()(image=image, mask=mask)

    assert transformed["image"].shape[1:] == (512, 512)
