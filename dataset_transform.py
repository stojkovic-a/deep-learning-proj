import os
import pathlib

from glob import glob
from PIL import Image
import config as conf


def resize(input_path, output_path, output_size):
    file_names = sorted(glob(input_path + "/**"))
    for fn in file_names:
        path = pathlib.Path(fn)
        image = Image.open(path)
        image_resized = image.resize(
            size=(output_size, output_size), resample=Image.Resampling.LANCZOS
        )
        image_resized.save(os.path.join(output_path, path.name))


if __name__ == "__main__":
    # resize("./dataset/archive/train/photos", conf.PHOTOS_PATH, 286)
    resize("./dataset/archive/train/sketches", conf.CONTOURS_PATH, 286)
