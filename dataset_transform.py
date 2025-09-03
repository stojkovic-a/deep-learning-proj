import os
import pathlib

from glob import glob
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from tqdm import tqdm
import cv2 as cv

import config as conf


def canny_edges(input_path, output_path, low_threshold=100, high_threshold=150):
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

    edges = cv.Canny(img, low_threshold, high_threshold)

    edge_img = Image.fromarray(edges)

    if output_path:
        edge_img.save(output_path)

    return edge_img


def invert(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_names = sorted(glob(os.path.join(input_path, "**")))
    for fn in tqdm(file_names):
        path = pathlib.Path(fn)
        image = Image.open(path)
        array = np.array(image).astype(np.uint8)
        array = np.where(array == 255, 0, 255)
        array = np.expand_dims(array, 2)
        array = array.repeat(3, 2)
        array = array.astype(np.uint8)
        image = Image.fromarray(array)
        image.save(os.path.join(output_path, path.name))


def get_contours(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_names = sorted(glob(os.path.join(input_path, "**")))
    for fn in tqdm(file_names):
        path = pathlib.Path(fn)
        image = Image.open(path)
        image = image.filter(ImageFilter.FIND_EDGES)
        image = image.filter(ImageFilter.BLUR)
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageOps.invert(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        image.save(os.path.join(output_path, path.name))
        # edges=ImageFilter.FIND_EDGES()
        # image=edges.filter(image)
        # blur=ImageFilter.BLUR()
        # image=blur.filter(image)
        # sharpen=ImageFilter.SHARPEN()
        # image=sharpen.filter(image)
        # image=ImageOps.invert(image)
        # image=ImageEnhance.Contrast().enhance(1.5)


def resize(input_path, output_path, output_size):
    os.makedirs(output_path, exist_ok=True)
    file_names = sorted(glob(input_path + "/**"))
    for fn in tqdm(file_names):
        path = pathlib.Path(fn)
        image = Image.open(path)
        image_resized = image.resize(
            size=(output_size, output_size), resample=Image.Resampling.LANCZOS
        )
        image_resized.save(os.path.join(output_path, path.name))


def pad_resize(input_path, output_path, output_size):
    os.makedirs(output_path, exist_ok=True)
    file_names = sorted(glob(input_path + "/**", recursive=True))

    for fn in tqdm(file_names):
        path = pathlib.Path(fn)

        try:
            image = Image.open(path).convert("RGB")
            arr = np.array(image)

            h, w, _ = arr.shape
            if h == w:
                square = arr
            else:
                side = max(h, w)
                pad_top = (side - h) // 2
                pad_bottom = side - h - pad_top
                pad_left = (side - w) // 2
                pad_right = side - w - pad_left

                square = np.pad(
                    arr,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="edge",
                )

            square_img = Image.fromarray(square)
            image_resized = square_img.resize(
                (output_size, output_size), resample=Image.Resampling.LANCZOS
            )

            image_resized.save(os.path.join(output_path, path.name))

        except Exception as e:
            print(f"Skipping {fn}: {e}")


if __name__ == "__main__":
    # resize("./dataset/archive/train/photos", conf.PHOTOS_PATH, 286)
    # resize("./dataset/archive/train/sketches", conf.CONTOURS_PATH, 286)
    # resize("./dataset/archive/val/photos", conf.PHOTOS_PATH_VAL, 256)
    # resize("./dataset/archive/val/sketches", conf.CONTOURS_PATH_VAL, 256)
    # resize("./dataset/archive/test/photos", conf.PHOTOS_PATH_TEST, 256)
    # resize("./dataset/archive/test/sketches", conf.CONTOURS_PATH_TEST, 256)
    # pad_resize(
    #     "./dataset/transfer learning/archive/photos", conf.PHOTOS_PATH_TRANSFER, 256
    # )
    # pad_resize(
    #     "./dataset/transfer learning/archive/sketches", conf.CONTOURS_PATH_TRANSFER, 256
    # )
    # get_contours(conf.PHOTOS_PATH_TEST, conf.CONTOURS_PATH_TEST)
    # resize("./dataset/test/images","./dataset/test/images",256)
    canny_edges(
        "./dataset/test/images/4.jpeg", "./dataset/test/contours_inverted/4.png"
    )
    invert("./dataset/test/contours_inverted", conf.CONTOURS_PATH_TEST)
