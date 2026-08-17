"""Download pretrained wrinkle-segmentation weights from Hugging Face Hub.

Replaces the manual "download from Google Drive / Dropbox, ask the author if the
link is dead" flow with a single reproducible command:

    python scripts/download_weights.py

Downloads from the Hugging Face model repo referenced by `DEFAULT_REPO_ID`
below (see scripts/convert_to_safetensors.py for how weights get uploaded
there). Pass --repo-id to point at a different repo if needed.
"""

import argparse
import logging
import os

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_REPO_ID = "rmsandu/ffhq-wrinkle-unet"
DEFAULT_FILENAME = "wrinkle_model.safetensors"
DEFAULT_DEST_DIR = "res/cp"


def download_weights(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    dest_dir: str = DEFAULT_DEST_DIR,
) -> str:
    """Downloads `filename` from the `repo_id` HF model repo into `dest_dir`.

    Idempotent: if the file already exists locally, huggingface_hub's cache
    will skip re-downloading it.
    """
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        logging.info("Weights already present at %s, skipping download.", dest_path)
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)
    logging.info("Downloading %s from %s ...", filename, repo_id)
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # hf_hub_download caches into ~/.cache/huggingface; place a copy where
    # app.py / train.py expect it (res/cp/).
    if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
        import shutil

        shutil.copy2(downloaded_path, dest_path)

    logging.info("Weights ready at %s", dest_path)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--dest-dir", default=DEFAULT_DEST_DIR)
    args = parser.parse_args()

    download_weights(
        repo_id=args.repo_id, filename=args.filename, dest_dir=args.dest_dir
    )
