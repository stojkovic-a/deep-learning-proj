import torch
import torch.nn as nn


def init_weights_normal(m: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.zeros_(m.bias.data)


class DownBlockCustom(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str,
        norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not norm
        )
        self.norm = None
        if norm:
            if norm_type == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == "spectral":
                self.norm = nn.utils.spectral_norm(self.c)
            else:
                raise Exception("Not implemented")
        self.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c(x)
        if self.norm is not None and self.norm != "spectral":
            x = self.norm(x)
        x = self.a(x)
        return x


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not norm
        )
        self.norm = None
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.a(x)
        return x


class UpBlockCustom(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = True,
        dropout: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = None
        if norm:
            self.norm = nn.InstanceNorm2d(out_channels)
        self.a = nn.ReLU(inplace=True)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.c(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.a(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = True,
        dropout: bool = False,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.c = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not norm
        )
        self.norm = None
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.a = nn.ReLU(inplace=True)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.a(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNetGeneratorCustom(nn.Module):
    """U-Net Generator:
    Encoder:  C64-C128-C256-C512-C512-C512-C512-C512
    Decoder:  C512-C1024-C1024-C1024-512-C256-C128-C3
    Conv: 4x4
    Stride: 2
    No Batch Norm on first decoder
    Leaky ReLU slope: 0.2
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        start_filters: int = 64,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.down1 = DownBlockCustom(
            in_channels, start_filters, "instance", norm=False
        )  # C64
        self.down2 = DownBlockCustom(
            start_filters, start_filters * 2, "instance", norm=True
        )  # C128
        self.down3 = DownBlockCustom(
            start_filters * 2, start_filters * 4, "instance", norm=True
        )  # C256
        self.down4 = DownBlockCustom(
            start_filters * 4, start_filters * 8, "instance", norm=True
        )  # C512
        self.down5 = DownBlockCustom(
            start_filters * 8, start_filters * 8, "instance", norm=True
        )  # C512
        self.down6 = DownBlockCustom(
            start_filters * 8, start_filters * 8, "instance", norm=True
        )  # C512
        self.down7 = DownBlockCustom(
            start_filters * 8, start_filters * 8, "instance", norm=True
        )  # C512
        self.down8 = DownBlockCustom(
            start_filters * 8, start_filters * 8, "instance", norm=False
        )  # C512

        self.up1 = UpBlockCustom(
            start_filters * 8, start_filters * 8, norm=True, dropout=True
        )  # d7
        self.up2 = UpBlockCustom(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=True
        )  # d6
        self.up3 = UpBlockCustom(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=True
        )  # d5
        self.up4 = UpBlockCustom(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=False
        )  # d4
        self.up5 = UpBlockCustom(
            start_filters * 8 * 2, start_filters * 4, norm=True, dropout=False
        )  # d3
        self.up6 = UpBlockCustom(
            start_filters * 4 * 2, start_filters * 2, norm=True, dropout=False
        )  # d2
        self.up7 = UpBlockCustom(
            start_filters * 2 * 2, start_filters, norm=True, dropout=False
        )  # d1

        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final = nn.Conv2d(
            start_filters * 2, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)  # (N, 64, 128, 128)
        d2 = self.down2(d1)  # (N, 128, 64, 64)
        d3 = self.down3(d2)  # (N, 256, 32, 32)
        d4 = self.down4(d3)  # (N, 512, 16, 16)
        d5 = self.down5(d4)  # (N, 512, 8, 8)
        d6 = self.down6(d5)  # (N, 512, 4, 4)
        d7 = self.down7(d6)  # (N, 512, 2, 2)
        d8 = self.down8(d7)  # (N, 512, 1, 1)

        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        out = self.final_upsample(u7)
        out = self.final(out)
        if self.use_tanh:
            out = self.tanh(out)
        return out


class UNetGenerator(nn.Module):
    """U-Net Generator:
    Encoder:  C64-C128-C256-C512-C512-C512-C512-C512
    Decoder:  C512-C1024-C1024-C1024-512-C256-C128-C3
    Conv: 4x4
    Stride: 2
    No Batch Norm on first decoder
    Leaky ReLU slope: 0.2
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        start_filters: int = 64,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.down1 = DownBlock(in_channels, start_filters, norm=False)  # C64
        self.down2 = DownBlock(start_filters, start_filters * 2, norm=True)  # C128
        self.down3 = DownBlock(start_filters * 2, start_filters * 4, norm=True)  # C256
        self.down4 = DownBlock(start_filters * 4, start_filters * 8, norm=True)  # C512
        self.down5 = DownBlock(start_filters * 8, start_filters * 8, norm=True)  # C512
        self.down6 = DownBlock(start_filters * 8, start_filters * 8, norm=True)  # C512
        self.down7 = DownBlock(start_filters * 8, start_filters * 8, norm=True)  # C512
        self.down8 = DownBlock(start_filters * 8, start_filters * 8, norm=False)  # C512

        self.up1 = UpBlock(
            start_filters * 8, start_filters * 8, norm=True, dropout=True
        )  # d7
        self.up2 = UpBlock(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=True
        )  # d6
        self.up3 = UpBlock(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=True
        )  # d5
        self.up4 = UpBlock(
            start_filters * 8 * 2, start_filters * 8, norm=True, dropout=False
        )  # d4
        self.up5 = UpBlock(
            start_filters * 8 * 2, start_filters * 4, norm=True, dropout=False
        )  # d3
        self.up6 = UpBlock(
            start_filters * 4 * 2, start_filters * 2, norm=True, dropout=False
        )  # d2
        self.up7 = UpBlock(
            start_filters * 2 * 2, start_filters, norm=True, dropout=False
        )  # d1

        self.final = nn.ConvTranspose2d(
            start_filters * 2, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)  # (N, 64, 128, 128)
        d2 = self.down2(d1)  # (N, 128, 64, 64)
        d3 = self.down3(d2)  # (N, 256, 32, 32)
        d4 = self.down4(d3)  # (N, 512, 16, 16)
        d5 = self.down5(d4)  # (N, 512, 8, 8)
        d6 = self.down6(d5)  # (N, 512, 4, 4)
        d7 = self.down7(d6)  # (N, 512, 2, 2)
        d8 = self.down8(d7)  # (N, 512, 1, 1)

        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        out = self.final(u7)
        if self.use_tanh:
            out = self.tanh(out)
        return out


class PatchDiscriminatorCustom(nn.Module):
    """70x70 Patch discriminator:
    Disriminator: C4-C64-C128-C256-C512-C1  (
    Conv: 4x4
    Stride: 2
    No Batch Norm on first layer
    Leaky ReLU slope: 0.2
    """

    def __init__(
        self,
        real_channels: int = 3,
        condition_channels: int = 3,
        start_filters: int = 64,
    ) -> None:
        super().__init__()
        channels = real_channels + condition_channels

        self.c1 = DownBlockCustom(
            channels,
            start_filters,
            "spectral",
            norm=False,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.c2 = DownBlockCustom(
            start_filters,
            start_filters * 2,
            "spectral",
            kernel_size=4,
            stride=2,
            norm=True,
        )
        self.c3 = DownBlockCustom(
            start_filters * 2,
            start_filters * 4,
            "spectral",
            kernel_size=4,
            stride=2,
            norm=True,
        )
        self.c4 = DownBlockCustom(
            start_filters * 4,
            start_filters * 8,
            "spectral",
            kernel_size=4,
            stride=1,
            norm=True,
        )
        self.final = nn.Conv2d(start_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        # No sigmoid

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, y], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.final(x)
        return x


class PatchDiscriminator(nn.Module):
    """70x70 Patch discriminator:
    Disriminator: C4-C64-C128-C256-C512-C1  (
    Conv: 4x4
    Stride: 2
    No Batch Norm on first layer
    Leaky ReLU slope: 0.2
    """

    def __init__(
        self,
        real_channels: int = 3,
        condition_channels: int = 3,
        start_filters: int = 64,
    ) -> None:
        super().__init__()
        channels = real_channels + condition_channels

        self.c1 = DownBlock(
            channels, start_filters, norm=False, kernel_size=4, stride=2, padding=1
        )
        self.c2 = DownBlock(
            start_filters, start_filters * 2, kernel_size=4, stride=2, norm=True
        )
        self.c3 = DownBlock(
            start_filters * 2, start_filters * 4, kernel_size=4, stride=2, norm=True
        )
        self.c4 = DownBlock(
            start_filters * 4, start_filters * 8, kernel_size=4, stride=1, norm=True
        )
        self.final = nn.Conv2d(start_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        # No sigmoid

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, y], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.final(x)
        return x
