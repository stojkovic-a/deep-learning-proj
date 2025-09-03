import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNetGenerator
from train import get_val_test_transforms
from dataset import ImageContourDataset
import config as conf


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    G = UNetGenerator(
        in_channels=conf.NUM_CONTOUR_CHANNELS,
        out_channels=conf.NUM_IMAGE_CHANNELS,
        start_filters=64,
    )
    G.to(device)
    ds = ImageContourDataset(
        conf.PHOTOS_PATH_TEST,
        conf.CONTOURS_PATH_TEST,
        get_val_test_transforms(),
    )
    batch_size = 1
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    state = torch.load(conf.STATE_PATH)
    G.load_state_dict(state["gen"])

    inference_path = conf.INFERENCE_PATH
    os.makedirs(inference_path, exist_ok=True)
    inference_path = os.path.join(inference_path, f"{len(os.listdir(inference_path))}")
    os.makedirs(inference_path, exist_ok=True)

    G.train()
    for i, (condition, image) in tqdm(enumerate(dataloader)):
        condition = condition.to(device)
        image = image.to(device)
        with torch.no_grad():
            fake = G(condition)

        image = torch.squeeze(image, 0)
        image = image * 255
        image = torch.clip(image, 0, 255)
        condition = torch.squeeze(condition, 0)
        condition = condition * 255
        condition = torch.clip(condition, 0, 255)
        fake = torch.squeeze(fake, 0)
        fake = fake * 255
        fake = torch.clip(fake, 0, 255)

        if condition.shape[0] == 1:
            condition = condition.repeat(3, 1, 1)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if fake.shape[0] == 1:
            fake = fake.repeat(3, 1, 1)

        combined = torch.cat([image, condition, fake], dim=2)
        combined = combined.permute(1, 2, 0)
        combined = combined.detach().cpu().numpy().astype(np.uint8)
        combined = Image.fromarray(combined)
        combined.save(os.path.join(inference_path, f"{i}.png"))
