# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Facial wrinkle segmentation using a ResNet50-encoder U-Net, trained on the FFHQ-Wrinkle dataset. The task has severe class imbalance (wrinkle pixels are ~0.03% of all pixels), which drives most of the design decisions in the loss functions, dataset handling, and evaluation metrics.

## Commands

All scripts are run directly from the repo root (referred to below as `ffhq-detect-wrinkles`).

```bash
pip install -r requirements.txt

# Dev/test tooling (pytest, ruff) — separate from the pinned training requirements
pip install -r requirements-dev.txt
pytest -v                # runs tests/ — see below
ruff check .              # lint is non-blocking in CI due to pre-existing debt in legacy files

# Train (reads hyperparameters from config.yaml, not CLI args)
python train.py

# Evaluate a trained checkpoint (imported as a function by train.py; see evaluate() signature for direct use)
python evaluate.py

# Fetch pretrained weights (once the HF model repo referenced in scripts/download_weights.py exists)
python scripts/download_weights.py

# Run the Gradio demo (requires res/cp/wrinkle_model.pth and res/cp/face_segmentation.pth).
# requirements-demo.txt is a lighter-weight subset of requirements.txt sufficient for just app.py.
python app.py

# Dataset prep pipeline (see README for the full folder-structure walkthrough)
bash download_ffhq_wrinkle.sh          # downloads manual + weak wrinkle masks
bash face_masking.sh                    # crops/masks face images to match wrinkle labels
python png_parsing.py <images1024x1024_dir> <manual_wrinkle_masks_dir> <face_images_dir>
python face_parsing_extraction.py       # BiSeNet-based face parsing/masking (needs face-parsing.PyTorch checkout + weights in res/cp/)
```

Training is configured entirely via `config.yaml` (epochs, batch size, LR, augmentation toggle, `use_attention`, `pretrained`, `freeze_encoder`, `dilate_mask`, checkpoint/data dirs). Edit that file rather than adding CLI args. Training uses [Weights & Biases](https://wandb.ai) for logging (`wandb_project`/`wandb_entity` in config.yaml) — `wandb.init()` runs unconditionally in `train_model`, so a wandb login is required to run training at all.

## Architecture

**Model (`unet/`)**: `UNet` (`unet_model.py`) is a U-Net with a **pretrained ResNet50 encoder** (`torchvision.models.resnet50`) and a custom decoder built from `Up`/`DoubleConv`/`OutConv` blocks (`unet_parts.py`). Encoder stages come from `resnet.layer1..4`; the decoder has 5 `Up` blocks that mirror them plus one extra upsampling stage back to full resolution. `use_attention=True` inserts `AttentionGate` modules (`unet_parts.py`) between corresponding encoder/decoder skip connections — when disabled, skip connections pass through unchanged. `freeze_encoder` toggles `requires_grad` on all ResNet50 parameters. Output is raw logits, shape `(B, 1, 512, 512)` for the single wrinkle class.

**Losses (`losses.py`)**: Training/eval loss is `binary_focal_loss(logits, target)` + `dice_loss(sigmoid_probs, target)` computed and logged separately, then summed — this combination specifically targets the extreme class imbalance (focal loss down-weights easy negatives; Dice loss is insensitive to the background/foreground ratio).

**Data pipeline (`utils/dataset_loading.py`)**: `WrinkleDataset` pairs sorted filenames from `image_dir`/`mask_dir` by matching stem names (order-dependent — image/mask directories must contain identically-named files). All images/masks are resized to 512×512 (LANCZOS for images, NEAREST for masks, to preserve binary mask values). Masks are binarized at threshold 127 and asserted to contain only `{0, 1}`. An optional `dilate_mask` step (3×3 kernel, `cv2.dilate`) thickens thin wrinkle lines before training — note `train.py` currently hardcodes `dilate_mask=False` for both train/val datasets regardless of `config.yaml`'s `dilate_mask` key (the config value is read into an unused local `dilate_mask` var). `get_augmentation_transforms()` vs `get_debug_transforms()` (both Albumentations pipelines, selected by `config["augmentation"]`) control whether rotation/flip/affine/CLAHE/blur/noise augmentation is applied; both end in ImageNet normalization + `ToTensorV2`.

**Training loop (`train.py`)**: Loads `config.yaml` → builds `WrinkleDataset` → fixed-seed (42) train/val split via `random_split` → AdamW + `ReduceLROnPlateau` (based on val loss) → per-epoch calls `evaluate.evaluate()` for validation metrics → saves `best_checkpoint.pth` (by val IoU), `last_checkpoint.pth` (every epoch), and `final_checkpoint.pth` (on exit/interrupt) under `config["checkpoint_dir"]`. Early stopping via `patience` (epochs without val IoU improvement). Checkpoints store `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and `all_metrics`.

**Evaluation (`evaluate.py`)**: `evaluate()` is called both from `train.py` (per-epoch validation) and standalone. Computes precision/recall/F1(=Dice)/AUROC/IoU via `torchmetrics`, plus averaged focal/dice loss. Can log overlay images (true mask = green, predicted mask = red) to W&B and/or save them locally under `results/{mode}/{run_name}/epoch_{epoch}/`.

**Inference/demo (`app.py`, `face_parsing_extraction.py`, `face_detection.py`)**: The Gradio demo pipeline is: Haar-cascade face detection (`face_detection.detect_face`) → BiSeNet face parsing to mask out everything but face+nose (`face_parsing_extraction.parse_face`, requires a separate `face-parsing.PyTorch` checkout on `sys.path` plus `res/cp/face_segmentation.pth`) → resize to 512×512, ImageNet-normalize → `UNet` forward pass → sigmoid + threshold at 0.5 → wrinkle percentage (`face_detection.calculate_wrinkle_metrics`) and overlay display. Both `.pth` checkpoints (`res/cp/wrinkle_model.pth`, `res/cp/face_segmentation.pth`) must be downloaded manually per the README before `app.py` will run; they are not in the repo.

## Tests (`tests/`)

CPU-only, no real checkpoints or FFHQ data needed — `UNet` is instantiated with `pretrained=False` in tests to avoid pulling torchvision hub weights. Covers `losses.py` (pure functions), `face_detection.calculate_wrinkle_metrics`, `WrinkleDataset` invariants (via synthetic fixture images in `tmp_path`), and shape smoke tests for `unet/unet_model.py` and `unet/unet_parts.py`. Config lives in `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]`). CI (`.github/workflows/ci.yml`) runs on push/PR against Python 3.10/3.11 with CPU torch wheels; lint failures don't block the build (pre-existing lint debt in files outside this pass — see `ruff check .` output).

## Notes for making changes

- `config.yaml` is the single source of truth for hyperparameters; if you add a new tunable, wire it through `load_config`/`train_model` in `train.py` rather than hardcoding it.
- `WrinkleDataset.__getitem__` has many `assert` statements enforcing tensor shape/value invariants (3-channel CHW images, single-channel masks with values in `{0,1}`, matching spatial dims) — preserve these if you touch the transform pipeline, since they're the main correctness guard for the imbalance-sensitive losses.
- Image/mask directories are paired by identical filenames (not by order alone) — `WrinkleDataset` sorts both lists but does not verify pairing beyond an assert on the file stem at load time.
- `UNet.forward` in `unet/unet_model.py` builds attention-gated skip connections (`self.att1`..`self.att4`) only when `use_attention=True`; those `AttentionGate` modules are constructed in `__init__` under the same flag — keep both in sync if you touch attention wiring (they were previously out of sync, causing a crash whenever `use_attention=True`, which is `config.yaml`'s default).
