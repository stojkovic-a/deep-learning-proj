import os
import pathlib

from glob import glob
from PIL import Image
from tqdm import tqdm

import config as conf


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


if __name__ == "__main__":
    # resize("./dataset/archive/train/photos", conf.PHOTOS_PATH, 286)
    # resize("./dataset/archive/train/sketches", conf.CONTOURS_PATH, 286)
    resize("./dataset/archive/val/photos", conf.PHOTOS_PATH_VAL, 256)
    resize("./dataset/archive/val/sketches", conf.CONTOURS_PATH_VAL, 256)
    resize("./dataset/archive/test/photos", conf.PHOTOS_PATH_TEST, 256)
    resize("./dataset/archive/test/sketches", conf.CONTOURS_PATH_TEST, 256)
