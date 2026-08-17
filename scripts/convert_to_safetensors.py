"""Convert a training checkpoint (.pth) to a safetensors weights file.

Training checkpoints saved by train.py are a dict with several keys
(model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch,
best_val_iou, all_metrics, ...). safetensors only stores tensors, so this
extracts just `model_state_dict` and writes it out as a .safetensors file,
carrying the scalar metrics over as string metadata (safetensors metadata
values must be strings).

Usage:
    python scripts/convert_to_safetensors.py best_checkpoint_iou032.pth \
        --output res/cp/wrinkle_model.safetensors

Optionally upload the converted file straight to the Hugging Face model repo:
    python scripts/convert_to_safetensors.py best_checkpoint_iou032.pth \
        --push --repo-id rmsandu/ffhq-wrinkle-unet
"""

import argparse
import logging
import os

import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_REPO_ID = "rmsandu/ffhq-wrinkle-unet"


def convert(checkpoint_path: str, output_path: str) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = (
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    )

    # safetensors requires contiguous, non-shared-storage CPU tensors.
    state_dict = {k: v.contiguous().clone() for k, v in state_dict.items()}

    metadata = {}
    if "epoch" in checkpoint:
        metadata["epoch"] = str(checkpoint["epoch"])
    if "best_val_iou" in checkpoint:
        metadata["best_val_iou"] = str(checkpoint["best_val_iou"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)
    logging.info(
        "Wrote %d tensors (%s) to %s", len(state_dict), metadata, output_path
    )
    return output_path


def push_to_hub(file_path: str, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="model",
    )
    logging.info("Uploaded %s to https://huggingface.co/%s", file_path, repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to the .pth checkpoint to convert")
    parser.add_argument(
        "--output",
        default="res/cp/wrinkle_model.safetensors",
        help="Where to write the converted .safetensors file",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the converted file to the Hugging Face model repo after converting",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo id to push to (only used with --push)",
    )
    args = parser.parse_args()

    output_path = convert(args.checkpoint, args.output)

    if args.push:
        push_to_hub(output_path, args.repo_id)
