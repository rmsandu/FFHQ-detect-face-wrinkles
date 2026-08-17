"""Download pretrained model weights from Hugging Face Hub.

Replaces the manual "download from Google Drive / Dropbox, ask the author if the
link is dead" flow with a single reproducible command:

    python scripts/download_weights.py

Downloads both the wrinkle-segmentation weights and the BiSeNet face-parsing
weights from the Hugging Face model repo referenced by `DEFAULT_REPO_ID` below
(see scripts/convert_to_safetensors.py for how weights get uploaded there).
Pass --repo-id to point at a different repo if needed. app.py calls
download_all_weights() automatically on startup if weights aren't present
locally (e.g. on a fresh Hugging Face Spaces deployment).
"""

import argparse
import logging
import os
import shutil

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_REPO_ID = "rmsandu/ffhq-wrinkle-unet"
WRINKLE_MODEL_FILENAME = "wrinkle_model.safetensors"
FACE_PARSING_FILENAME = "face_segmentation.safetensors"
DEFAULT_DEST_DIR = "res/cp"


def download_weights(
    filename: str,
    repo_id: str = DEFAULT_REPO_ID,
    dest_dir: str = DEFAULT_DEST_DIR,
) -> str:
    """Downloads `filename` from the `repo_id` HF model repo into `dest_dir`.

    Idempotent: if the file already exists locally, this is a no-op.
    """
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        logging.info("Weights already present at %s, skipping download.", dest_path)
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)
    logging.info("Downloading %s from %s ...", filename, repo_id)
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # hf_hub_download caches into ~/.cache/huggingface; place a copy where
    # app.py / train.py / face_parsing_extraction.py expect it (res/cp/).
    if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
        shutil.copy2(downloaded_path, dest_path)

    logging.info("Weights ready at %s", dest_path)
    return dest_path


def download_all_weights(
    repo_id: str = DEFAULT_REPO_ID, dest_dir: str = DEFAULT_DEST_DIR
) -> tuple[str, str]:
    """Downloads both the wrinkle-segmentation and face-parsing weights."""
    wrinkle_path = download_weights(WRINKLE_MODEL_FILENAME, repo_id, dest_dir)
    face_parsing_path = download_weights(FACE_PARSING_FILENAME, repo_id, dest_dir)
    return wrinkle_path, face_parsing_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dest-dir", default=DEFAULT_DEST_DIR)
    args = parser.parse_args()

    download_all_weights(repo_id=args.repo_id, dest_dir=args.dest_dir)
