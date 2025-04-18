from datetime import datetime
import gradio as gr
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from unet import UNet
from dotenv import load_dotenv
from face_parsing_extraction import parse_face
from face_detection import detect_face, calculate_wrinkle_metrics
from unet.unet_parts import Up

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

checkpoint = torch.load("res/cp/wrinkle_model.pth", map_location=device)
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

model.load_state_dict(checkpoint["model_state_dict"])  # <- shapes now match

print("Model loaded successfully.")
# Preprocessing transformation
wrinkle_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def preprocess_and_predict(
    img: Image.Image,
) -> np.ndarray:
    """Process the resized image and generate wrinkle overlay."""
    # create a new directory for the image and save it there with a unique timestap and name

    if img is None:
        gr.Warning("No image uploaded. Please upload an image to proceed.")
        raise ValueError("No image provided!")

    resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)

    # check if there is a human face in the uploaded image, otherwise display a warning with Gradio
    face_detected = detect_face(resized_img)
    if face_detected is None:
        gr.Warning(
            "No human face detected. Please upload a photo with a close-up shot of a face."
        )
        # raise ValueError("No face detected!")

    image_dir = "output_images"
    os.makedirs(image_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # make subdirectory for the image
    os.makedirs(f"{image_dir}/{timestamp}", exist_ok=True)
    sub_dir_path = f"{image_dir}/{timestamp}"
    resized_img.save(os.path.join(sub_dir_path, f"image_{timestamp}.png"))
    processed_face = parse_face(dspth=sub_dir_path, cp="face_segmentation.pth")
    # Wrinkle detection
    face_tensor = wrinkle_transform(processed_face).unsqueeze(0).to(device)
    with torch.no_grad():
        wrinkle_output = model(face_tensor)
        wrinkle_prediction = torch.sigmoid(wrinkle_output).cpu().numpy()
    # display warning if wrinkle prediction is empty
    if wrinkle_prediction.size == 0:
        gr.Warning("No wrinkle prediction found. YOU ARE PERFECT.")
    # Create overlay

    wrinkle_mask = (wrinkle_prediction > 0.5).astype(np.uint8)
    wrinkle_percentage_unet = calculate_wrinkle_metrics(wrinkle_mask)

    annotations = [
        (wrinkle_mask, "Segmentation Wrinkles"),  # Label for DL mask
    ]

    return (
        (resized_img, annotations),
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
            run_button = gr.Button("Run Model", variant="primary")

            wrinkle_percentage_unet = gr.Label(label="Wrinkle Percentage (UNet):")

            # Run the model
            run_button.click(
                fn=preprocess_and_predict,
                inputs=[input_image],
                outputs=[
                    result_image,
                    wrinkle_percentage_unet,
                ],
            )


# Authentication and Launch
demo.queue().launch()
