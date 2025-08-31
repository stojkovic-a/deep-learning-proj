import os

from glob import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class ImageContourDataset(Dataset):
    def __init__(self, image_path, contour_path, transforms):
        self.transforms = transforms
        self.image_path = image_path
        self.contour_path = contour_path

        self.image_files = sorted(glob(os.path.join(self.image_path, "**")))
        self.contour_files = sorted(glob(os.path.join(self.contour_path, "**")))

        assert len(self.image_files) > 0
        assert len(self.image_files) == len(self.contour_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        contour = np.array(Image.open(self.contour_files[idx]).convert("L"))
        contour = np.expand_dims(contour, -1)
        image = np.array(Image.open(self.image_files[idx]).convert("RGB"))

        if self.transforms:
            image, contour = self.transforms((image, contour))
        return contour, image
