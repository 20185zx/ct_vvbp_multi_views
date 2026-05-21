import torch
import torch.nn as nn

from src.geometry.fbp import LInFBPFixedLinearFBPBatch


def periodic_view_upsample(sino, target_views: int):
    """
    Periodic linear interpolation along the view dimension.

    Input:
        sino: [B, 1, V_in, D]
    Output:
        out:  [B, 1, target_views, D]

    This treats the angular dimension as periodic, so the last sparse view
    is interpolated with the first sparse view near 2*pi.
    """
    B, C, V_in, D = sino.shape
    device = sino.device

    if target_views == V_in:
        return sino

    # Output angle index mapped to sparse-view coordinate.
    pos = torch.arange(target_views, device=device, dtype=torch.float32) * (V_in / target_views)

    idx0 = torch.floor(pos).long() % V_in
    idx1 = (idx0 + 1) % V_in
    frac = (pos - torch.floor(pos)).view(1, 1, target_views, 1)

    y0 = sino[:, :, idx0, :]
    y1 = sino[:, :, idx1, :]

    out = (1.0 - frac) * y0 + frac * y1
    return out


class SinoResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class SinoUpsampleResidualFBPNet(nn.Module):
    """
    Sparse-to-full-view sinogram residual baseline + fixed full-view FBP.

    Input:
        sparse sino: [B, 1, V_in, D]

    Process:
        1. Periodic linear interpolation: V_in → V_full views
        2. Sinogram-domain residual correction
        3. Fixed full-view FBP

    Output:
        recon image: [B, 1, H, W]
    """

    def __init__(
        self,
        geo_full,
        input_views=60,
        full_views=240,
        sino_channels=32,
        num_blocks=5,
        residual_scale=0.1,
        use_sino_norm=True,
    ):
        super().__init__()

        self.input_views = input_views
        self.full_views = full_views
        self.residual_scale = residual_scale
        self.use_sino_norm = use_sino_norm

        self.fbp = LInFBPFixedLinearFBPBatch(geo_full)

        self.entry = nn.Sequential(
            nn.Conv2d(1, sino_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[SinoResidualBlock(sino_channels) for _ in range(num_blocks)]
        )

        self.exit = nn.Conv2d(sino_channels, 1, kernel_size=3, padding=1)

    def forward(self, sino_sparse):
        """
        sino_sparse: [B, 1, V_in, D]
        """
        sino_full_base = periodic_view_upsample(sino_sparse, self.full_views)

        if self.use_sino_norm:
            mean = sino_full_base.mean(dim=(2, 3), keepdim=True)
            std = sino_full_base.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

            x = (sino_full_base - mean) / std
            residual = self.exit(self.body(self.entry(x)))

            sino_full_corr = sino_full_base + self.residual_scale * residual * std
        else:
            residual = self.exit(self.body(self.entry(sino_full_base)))
            sino_full_corr = sino_full_base + self.residual_scale * residual

        recon = self.fbp(sino_full_corr)

        return recon