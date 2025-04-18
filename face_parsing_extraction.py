#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Module for face parsing using BiSeNet.
"""
import sys
import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# Add the face-parsing.PyTorch directory to the Python path
face_parsing_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "face-parsing.PyTorch"
)
sys.path.append(face_parsing_dir)

from model import BiSeNet


def parse_face(
    respth="./res/test_res",
    dspth="./data",
    cp="face_segmentation.pth",
):
    """
    :param respth: path to save the result if needed
    :param dspth: path to a directory of images for processing /
    :param cp: checkpoint file (weights)"""

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join("res/cp", cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.Resampling.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # display parsing as an image
            # plt.imsave(
            #    osp.join(respth, "labels.png"),
            #    parsing,
            #    cmap="tab20",)
            # Crop the excluded labels
            exclude_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]
            # Crop the excluded labels
            exclude_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]
            parsing_anno = np.where(np.isin(parsing, exclude_labels), 0, parsing)

            # apply the mask to the original image
            im = np.array(image)
            masked_image = im * (parsing_anno[:, :, np.newaxis] > 0)
            result_image = Image.fromarray(masked_image.astype("uint8")).convert("RGB")
            # save the result image in the respth with the original filename
            filename_img = image_path.split(".")[0]
            result_image.save(osp.join(respth, f"{filename_img}.png"))

    return result_image


if __name__ == "__main__":
    # apply the mask to the original image and save the result into a folder
    # with the same name as the image
    BASE_FOLDER = "data/face_images/"
    MASKED_FACE_IMAGES = "data/masked_face_images/"

    # apply the mask to the original image and save the result into a folder with the same name as the image
    # create new masked face images folder if doesn't exist
    if not os.path.exists(MASKED_FACE_IMAGES):
        os.makedirs(MASKED_FACE_IMAGES)
    parse_face(respth=MASKED_FACE_IMAGES, dspth=BASE_FOLDER, cp="face_segmentation.pth")
    if not os.path.exists(MASKED_FACE_IMAGES):
        os.makedirs(MASKED_FACE_IMAGES)
    parse_face(respth=MASKED_FACE_IMAGES, dspth=BASE_FOLDER, cp="face_segmentation.pth")

    print("Face parsing completed.")
