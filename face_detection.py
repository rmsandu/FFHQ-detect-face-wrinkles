import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def calculate_wrinkle_metrics(mask: np.ndarray) -> tuple[float, int]:
    """
    Calculate wrinkle metrics: percentage of pixels and wrinkle count.

    Args:
        mask (np.ndarray): Binary wrinkle mask (0 and 1).

    Returns:
        float: Wrinkle percentage relative to total pixels.
        int: Wrinkle count (number of connected wrinkle regions).
    """
    mask = mask.astype(np.uint8)
    if mask.ndim == 4:  # If the mask has shape (1, 1, H, W)
        mask = np.squeeze(mask, axis=(0, 1))
    elif mask.ndim == 3:  # If the mask has shape (1, H, W)
        mask = np.squeeze(mask, axis=0)
    wrinkle_percentage = (np.sum(mask > 0) / mask.size) * 100

    # Calculate wrinkle count using connected components

    return round(wrinkle_percentage, 2)


def detect_face(img: Image.Image):
    # Convert the PIL image to a NumPy array
    img_np = np.array(img.convert("RGB"))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Check if any faces are detected
    if len(faces) == 0:
        return None

    # Draw rectangles around detected faces
    for x, y, w, h in faces:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the NumPy array back to a PIL image
    result_img = Image.fromarray(img_np)

    return result_img
