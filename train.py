import os

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import PatchDiscriminator, UNetGenerator, init_weights_normal
from dataset import ImageContourDataset
import config as conf


def build_pix2pix(
    image_channels: int = 3,
    condition_channels: int = 1,
    out_channels: int = 3,
    start_filters_gen: int = 64,
    start_filters_disc: int = 64,
):
    G = UNetGenerator(
        in_channels=condition_channels,
        out_channels=out_channels,
        start_filters=start_filters_gen,
    )
    D = PatchDiscriminator(
        real_channels=image_channels,
        condition_channels=condition_channels,
        start_filters=start_filters_disc,
    )
    G.apply(init_weights_normal)
    D.apply(init_weights_normal)
    return G, D


def get_gen_loss(g, d, real, condition, recon_crit, adv_crit, lambda_recon, lambda_adv):
    fake = g(condition)
    # with torch.no_grad():
    # disc_fake = d(fake, condition)
    disc_fake = d(fake, condition)
    adv_loss = adv_crit(disc_fake, torch.ones_like(disc_fake))
    rec_loss = recon_crit(real, fake)
    # print(adv_loss.item())
    # print(rec_loss.item())
    gen_loss = adv_loss * lambda_adv + rec_loss * lambda_recon
    return gen_loss, (adv_loss * lambda_adv).item(), (rec_loss * lambda_recon).item()


def get_train_transforms(crop_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.ToTensor(),
            transforms.RandomCrop(size=(crop_size, crop_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ToTensor(),
        ]
    )


def get_val_test_transforms():
    return transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )


def save_checkpoint(checkpoint_path, G, D, g_optim, d_optim):
    num_dirs = len(os.listdir(checkpoint_path))
    save_dir = os.path.join(checkpoint_path, f"{num_dirs}")
    os.makedirs(save_dir)
    torch.save(
        {
            "gen": G.state_dict(),
            "disc": D.state_dict(),
            "gen_opt": g_optim.state_dict(),
            "disc_opt": d_optim.state_dict(),
        },
        save_dir + f"/{len(os.listdir(save_dir))}.pth",
    )


def save_results(G, D, result_path, test_images, test_contours, device):
    G.eval()
    D.eval()
    num_dirs = len(os.listdir(result_path))
    save_dir = os.path.join(result_path, f"{num_dirs}")
    os.makedirs(save_dir)
    for m in G.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    for i, (img, contour) in enumerate(zip(test_images, test_contours)):
        img = img.to(device)
        contour = contour.to(device)
        with torch.no_grad():
            fake = G(torch.unsqueeze(contour, 0))

        img_cpu = img.detach().cpu()
        contour_cpu = contour.detach().cpu()
        fake_cpu = fake.detach().cpu().squeeze()
        if contour_cpu.shape[0] == 1:
            contour_cpu = contour_cpu.repeat(3, 1, 1)

        if img_cpu.shape[0] == 1:
            img_cpu = img_cpu.repeat(3, 1, 1)

        if fake_cpu.shape[0] == 1:
            fake_cpu = fake_cpu.repeat(3, 1, 1)
        combined = torch.cat([img_cpu, contour_cpu, fake_cpu], dim=2)
        vutils.save_image(combined, os.path.join(save_dir, f"{i}.png"), normalize=True)
    G.train()
    D.train()


def save_val(
    G,
    D,
    val_path,
    val_dataloader,
    device,
    val_losses,
    rec_crit,
    adv_crit,
    lambda_rec,
    lambda_adv,
):
    G.eval()
    D.eval()
    mean_gen_adv_loss = 0
    mean_gen_rec_loss = 0
    mean_disc_loss = 0
    num_dirs = len(os.listdir(val_path))
    save_dir = os.path.join(val_path, f"{num_dirs}")
    num_images = len(val_dataloader)

    os.makedirs(save_dir, exist_ok=True)
    for m in G.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    for i, (condition, image) in enumerate(val_dataloader):
        image = image.to(device)
        condition = condition.to(device)
        with torch.no_grad():
            fake = G(condition)
            _, adv_loss, rec_loss = get_gen_loss(
                G, D, image, condition, rec_crit, adv_crit, lambda_rec, lambda_adv
            )
            disc_fake = D(fake, condition)
            disc_fake_loss = adv_crit(disc_fake, torch.zeros_like(disc_fake))
            disc_real = D(image, condition)
            disc_real_loss = adv_crit(disc_real, torch.ones_like(disc_real))
            disc_loss = 0.5 * (disc_fake_loss + disc_real_loss)
            mean_gen_adv_loss += adv_loss / num_images
            mean_gen_rec_loss += rec_loss / num_images
            mean_disc_loss += disc_loss.item() / num_images
            img_cpu = image.detach().cpu()
            condition_cpu = condition.detach().cpu()
            fake_cpu = fake.detach().cpu()
            if condition_cpu.shape[1] == 1:
                condition_cpu = condition_cpu.repeat(1, 3, 1, 1)

            if img_cpu.shape[1] == 1:
                img_cpu = img_cpu.repeat(1, 3, 1, 1)

            if fake_cpu.shape[1] == 1:
                fake_cpu = fake_cpu.repeat(1, 3, 1, 1)

            combined = torch.cat([img_cpu, condition_cpu, fake_cpu], dim=3)
            vutils.save_image(
                combined, os.path.join(save_dir, f"{i}.png"), normalize=True
            )
    val_losses.append((mean_gen_adv_loss, mean_gen_rec_loss, mean_disc_loss))
    plt.figure(figsize=(8, 6))
    plt.plot(
        [x[0] for x in val_losses],
        label="Generator Adversarial Loss",
        color="red",
    )
    plt.plot(
        [x[1] for x in val_losses],
        label="Generator Reconstruction Loss",
        color="blue",
    )
    plt.plot([x[2] for x in val_losses], label="Discriminator Loss", color="green")
    plt.xlabel("Save number")
    plt.ylabel("Losses")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    os.makedirs(val_path, exist_ok=True)
    save_path = os.path.join(val_path, "loss_val.png")
    plt.savefig(save_path)
    plt.close()
    G.train()
    D.train()
    return val_losses


if __name__ == "__main__":
    updates_per_result = conf.UPDATES_PER_RESULT
    updates_per_checkpoint = conf.UPDATES_PER_CHECKPOINT
    updates_per_val = conf.UPDATES_PER_VAL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    G, D = build_pix2pix(
        image_channels=conf.NUM_IMAGE_CHANNELS,
        condition_channels=conf.NUM_CONTOUR_CHANNELS,
        out_channels=conf.NUM_IMAGE_CHANNELS,
    )
    G.to(device)
    D.to(device)
    ds = ImageContourDataset(
        conf.PHOTOS_PATH, conf.CONTOURS_PATH, get_train_transforms(conf.IMAGE_DIM)
    )
    # ds = ImageContourDataset(
    #     conf.PHOTOS_PATH_TRANSFER,
    #     conf.CONTOURS_PATH_TRANSFER,
    #     get_train_transforms(conf.IMAGE_DIM),
    # )
    batch_size = conf.BATCH_SIZE
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    val_ds = ImageContourDataset(
        conf.PHOTOS_PATH_VAL,
        conf.CONTOURS_PATH_VAL,
        transforms=get_val_test_transforms(),
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    reconstruction_criterion = nn.L1Loss()
    adversarial_criterion = nn.BCEWithLogitsLoss()
    lambda_rec = conf.LAMBDA_L1
    lambda_adv = conf.LAMBDA_ADVERSARIAL
    gen_lr = conf.LR_GEN
    disc_lr = conf.LR_DISC
    num_epoches = conf.NUM_EPOCHES
    betas = conf.BETAS
    g_optim = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=betas)
    d_optim = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=betas)

    if conf.PRETRAINED:
        print("check")
        state = torch.load(conf.STATE_PATH)
        G.load_state_dict(state["gen"])
        D.load_state_dict(state["disc"])
        g_optim.load_state_dict(state["gen_opt"])
        d_optim.load_state_dict(state["disc_opt"])

    checkpoint_path = conf.CHECKPOINT_PATH
    result_path = conf.RESULTS_PATH
    val_path = conf.VAL_PATH
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    val_path = os.path.join(val_path, f"{len(os.listdir(val_path))}")
    os.makedirs(val_path, exist_ok=True)

    G.train()
    D.train()
    step = 1
    test_images = []
    test_contours = []
    for i in range(50):
        temp = ds.__getitem__(i)
        test_images.append(temp[1])
        test_contours.append(temp[0])
    mean_gen_adv_loss = 0
    mean_gen_rec_loss = 0
    mean_disc_loss = 0
    graph_gen_adv_losses = []
    graph_gen_rec_losses = []
    graph_disc_losses = []
    val_losses = []
    for epoch in range(num_epoches):
        for condition, image in tqdm(dataloader):
            condition = condition.to(device)
            image = image.to(device)

            # Discriminator update
            with torch.no_grad():
                fake = G(condition)
            d_optim.zero_grad()
            disc_fake = D(fake, condition)
            disc_fake_loss = adversarial_criterion(
                disc_fake, torch.zeros_like(disc_fake)
            )
            disc_real = D(image, condition)
            disc_real_loss = adversarial_criterion(
                disc_real, torch.ones_like(disc_real)
            )
            disc_loss = 0.5 * (
                disc_fake_loss + disc_real_loss
            )  # U rad kazu da su ovo radili da uspore diskriminator
            # print(disc_loss.item())
            mean_disc_loss += disc_loss.item() / updates_per_val
            disc_loss.backward()
            d_optim.step()

            # Generator update
            g_optim.zero_grad()
            gen_loss, adv_loss_value, rec_loss_value = get_gen_loss(
                G,
                D,
                image,
                condition,
                reconstruction_criterion,
                adversarial_criterion,
                lambda_rec,
                lambda_adv,
            )
            mean_gen_adv_loss += adv_loss_value / updates_per_val
            mean_gen_rec_loss += rec_loss_value / updates_per_val
            # print(gen_loss.item())
            gen_loss.backward()
            g_optim.step()

            if step % updates_per_result == 0:
                save_results(G, D, result_path, test_images, test_contours, device)
            if step % updates_per_checkpoint == 0:
                save_checkpoint(checkpoint_path, G, D, g_optim, d_optim)
            if step % updates_per_val == 0:
                val_losses = save_val(
                    G,
                    D,
                    val_path,
                    val_dataloader,
                    device,
                    val_losses,
                    reconstruction_criterion,
                    adversarial_criterion,
                    lambda_rec,
                    lambda_adv,
                )
                graph_gen_adv_losses.append(mean_gen_adv_loss)
                graph_gen_rec_losses.append(mean_gen_rec_loss)
                graph_disc_losses.append(mean_disc_loss)

                plt.figure(figsize=(8, 6))
                plt.plot(
                    graph_gen_adv_losses,
                    label="Generator Adversarial Loss",
                    color="red",
                )
                plt.plot(
                    graph_gen_rec_losses,
                    label="Generator Reconstruction Loss",
                    color="blue",
                )
                plt.plot(graph_disc_losses, label="Discriminator Loss", color="green")
                plt.xlabel("Save number")
                plt.ylabel("Losses")
                plt.title("Training Loss Curves")
                plt.legend()
                plt.grid(True)
                os.makedirs(val_path, exist_ok=True)
                save_path = os.path.join(val_path, "loss.png")
                plt.savefig(save_path)
                plt.close()

                mean_gen_adv_loss = 0
                mean_gen_rec_loss = 0
                mean_disc_loss = 0

            if step == max(
                [updates_per_checkpoint, updates_per_result, updates_per_val]
            ):
                step = 0
            step += 1
