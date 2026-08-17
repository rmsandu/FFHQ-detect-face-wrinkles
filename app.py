from datetime import datetime
import logging
import shutil
import tempfile
import gradio as gr
import torch
import os
import cv2
import matplotlib.cm as cm
import numpy as np
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
from PIL import Image
from unet import UNet
from dotenv import load_dotenv
from face_parsing_extraction import parse_face
from face_detection import detect_face, calculate_wrinkle_metrics
from unet.unet_parts import Up
from scripts.download_weights import (
    FACE_PARSING_FILENAME,
    WRINKLE_MODEL_FILENAME,
    download_all_weights,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Uploaded images are processed in a temp dir and discarded by default. Set
# SAVE_UPLOADS=true to persist them under output_images/ for local debugging
# (do not enable this on a public deployment without a cleanup policy).
SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "false").lower() == "true"

# ---------------------------
# Pre-load models and settings
# ---------------------------

load_dotenv()

# Example images
example_dir = "example_images/"
example_images = [
    os.path.join(example_dir, img)
    for img in os.listdir(example_dir)
    if img.lower().endswith(("png", "jpg", "jpeg"))
]

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = os.path.join("res/cp", WRINKLE_MODEL_FILENAME)
FACE_PARSING_CHECKPOINT_PATH = os.path.join("res/cp", FACE_PARSING_FILENAME)

if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(FACE_PARSING_CHECKPOINT_PATH):
    try:
        logging.info("Model weights not found locally, downloading from Hugging Face Hub...")
        download_all_weights()
    except Exception as exc:
        raise FileNotFoundError(
            f"Wrinkle model checkpoint not found at '{CHECKPOINT_PATH}' and automatic "
            f"download failed ({exc}). Run `python scripts/download_weights.py` manually "
            "to fetch the pretrained weights before starting the demo."
        ) from exc

state_dict = load_safetensors(CHECKPOINT_PATH, device=device)
model = (
    UNet(
        n_channels=3,
        n_classes=1,
        bilinear=False,
        pretrained=True,
        freeze_encoder=True,
    )
    .to(device)
    .eval()
)

model.load_state_dict(state_dict)  # <- shapes now match

logging.info("Model loaded successfully from %s", CHECKPOINT_PATH)
# Preprocessing transformation
wrinkle_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def _wrinkle_probability_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Render a sigmoid probability map (H, W) as an RGB heatmap image."""
    colored = cm.get_cmap("inferno")(prob_map)  # (H, W, 4) RGBA in [0, 1]
    return (colored[..., :3] * 255).astype(np.uint8)


def preprocess_and_predict(
    img: Image.Image,
) -> np.ndarray:
    """Process the resized image and generate wrinkle overlay + confidence heatmap."""
    if img is None:
        gr.Warning("No image uploaded. Please upload an image to proceed.")
        raise gr.Error("No image provided! Please upload a photo to proceed.")

    resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)

    # check if there is a human face in the uploaded image, otherwise display a warning with Gradio
    face_detected = detect_face(resized_img)
    if face_detected is None:
        gr.Warning(
            "No human face detected. Please upload a photo with a close-up shot of a face."
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        with tempfile.TemporaryDirectory() as sub_dir_path:
            resized_img.save(os.path.join(sub_dir_path, f"image_{timestamp}.png"))
            processed_face = parse_face(dspth=sub_dir_path)

            if SAVE_UPLOADS:
                save_dir = os.path.join("output_images", timestamp)
                os.makedirs(save_dir, exist_ok=True)
                shutil.copytree(sub_dir_path, save_dir, dirs_exist_ok=True)

        # Wrinkle detection
        face_tensor = wrinkle_transform(processed_face).unsqueeze(0).to(device)
        with torch.no_grad():
            wrinkle_output = model(face_tensor)
            wrinkle_prediction = torch.sigmoid(wrinkle_output).cpu().numpy()
    except Exception as exc:
        logging.exception("Inference pipeline failed")
        raise gr.Error(f"Wrinkle detection failed: {exc}") from exc

    if wrinkle_prediction.size == 0:
        gr.Warning("No wrinkle prediction found. YOU ARE PERFECT.")

    prob_map = wrinkle_prediction[0, 0]  # (H, W) in [0, 1]
    wrinkle_mask = (prob_map > 0.5).astype(np.uint8)
    wrinkle_percentage_unet = calculate_wrinkle_metrics(wrinkle_mask)
    heatmap = _wrinkle_probability_heatmap(prob_map)

    annotations = [
        (wrinkle_mask, "Segmentation Wrinkles"),  # Label for DL mask
    ]

    return (
        (resized_img, annotations),
        heatmap,
        wrinkle_percentage_unet,
    )


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    # Add instructions for the user
    gr.Markdown(
        "This demo detects wrinkles using semantic segmentation. Upload a close-up photo of a face or select an example image to get started."
    )

    wrinkle_overlay = gr.State(None)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                image_mode="RGB",
                height=512,
                width=512,
                sources=["upload", "webcam", "clipboard"],
            )

            gr.Markdown("### Example Images")
            example_gallery = gr.Gallery(
                value=example_images,
                label="Example Images",
                columns=[6],
                rows=[1],
                object_fit="contain",
                height=200,
            )

            # Select an example image
            def get_select_index(evt: gr.SelectData) -> Image.Image:
                img_idx = evt.index
                return Image.open(example_images[img_idx])

            example_gallery.select(
                fn=get_select_index,
                inputs=None,
                outputs=input_image,
            )

        # Column 2: Output results
        with gr.Column():
            result_image = gr.AnnotatedImage(
                color_map={"Wrinkles": "#0000FF"},
                label="Wrinkle Detection Overlay",
            )
            confidence_heatmap = gr.Image(
                label="Model Confidence Heatmap",
                type="numpy",
            )
            run_button = gr.Button("Run Model", variant="primary")

            wrinkle_percentage_unet = gr.Label(label="Wrinkle Percentage (UNet):")

            # Run the model
            run_button.click(
                fn=preprocess_and_predict,
                inputs=[input_image],
                outputs=[
                    result_image,
                    confidence_heatmap,
                    wrinkle_percentage_unet,
                ],
            )


# Authentication and Launch
demo.queue().launch()
