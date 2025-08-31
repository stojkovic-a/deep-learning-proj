import os

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm

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
    return gen_loss


def get_transforms(crop_size: int):
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


if __name__ == "__main__":
    updates_per_result = conf.UPDATES_PER_RESULT
    updates_per_checkpoint = conf.UPDATES_PER_CHECKPOINT
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
        conf.PHOTOS_PATH, conf.CONTOURS_PATH, get_transforms(conf.IMAGE_DIM)
    )
    batch_size = conf.BATCH_SIZE
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
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
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    G.train()
    D.train()
    step = 0
    test_images = []
    test_contours = []
    for i in range(50):
        temp = ds.__getitem__(i)
        test_images.append(temp[1])
        test_contours.append(temp[0])

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
            disc_loss.backward()
            d_optim.step()

            # Generator update
            g_optim.zero_grad()
            gen_loss = get_gen_loss(
                G,
                D,
                image,
                condition,
                reconstruction_criterion,   
                adversarial_criterion,
                lambda_rec,
                lambda_adv,
            )
            # print(gen_loss.item())
            gen_loss.backward()
            g_optim.step()
            if step % updates_per_result == 0:
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
                    vutils.save_image(
                        combined, os.path.join(save_dir, f"{i}.png"), normalize=True
                    )
                    G.train()
                    D.train()

            if step % updates_per_checkpoint == 0:
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
                step = 0

            step += 1
    # for p in ds:
    #     print(p[0].shape, p[1].shape)
    #     break

    # x = torch.randn(2, 3, 256, 256, device=device)
    # with torch.no_grad():
    #     y_fake = G(x)
    # logits = D(x, y_fake)

    # print("G output:", tuple(y_fake.shape))  # (2, 3, 256, 256)
    # print("D output:", tuple(logits.shape))  # (2, 1, 30, 30)

    # # Example losses (placeholders)
    # criterion_gan = nn.BCEWithLogitsLoss()
    # criterion_l1 = nn.L1Loss()

    # # Targets for adversarial loss maps
    # real = torch.ones_like(logits)
    # fake = torch.zeros_like(logits)

    # # Generator adversarial + L1 (lambda=100 per paper)
    # pred_fake = D(x, y_fake)
    # loss_g_gan = criterion_gan(pred_fake, real)
    # # Dummy target y_real for illustration
    # y_real = torch.randn_like(y_fake)
    # loss_l1 = criterion_l1(y_fake, y_real) * 100.0
    # loss_g = loss_g_gan + loss_l1

    # # Discriminator loss
    # with torch.no_grad():
    #     y_fake_detached = G(x)
    # pred_real = D(x, y_real)
    # pred_fake_detached = D(x, y_fake_detached)
    # loss_d_real = criterion_gan(pred_real, real)
    # loss_d_fake = criterion_gan(pred_fake_detached, fake)
    # loss_d = 0.5 * (loss_d_real + loss_d_fake)

    # print(f"loss_g={loss_g.item():.4f} loss_d={loss_d.item():.4f}")
