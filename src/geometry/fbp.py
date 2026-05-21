import numpy as np
import torch
import torch.nn as nn


class LInFBPFixedLinearFBPBatch(nn.Module):
    """Batch-compatible fixed linear FBP layer.

    Input: sino [B, 1, views, nDetecU]
    Output: image [B, 1, 512, 512]
    """
    def __init__(self, geo):
        super().__init__()
        self.geo = geo

    def _filter_sinogram(self, sino):
        B, C, V, D = sino.shape
        npad = int(2 ** np.ceil(np.log2(2 * self.geo["nDetecU"] - 1)))
        pad = sino.new_zeros(B, C, V, npad - D)
        sino_pad = torch.cat([sino, pad], dim=-1)
        proj_fft = torch.fft.rfft(sino_pad, dim=-1, norm="ortho")
        filt = self.geo["filter"][: npad // 2 + 1].to(sino.device).view(1, 1, 1, -1)
        filtered_fft = proj_fft * filt
        filtered = torch.fft.irfft(filtered_fft, n=npad, dim=-1, norm="ortho")
        return filtered[..., :D]

    def _linear_backproject(self, sino_filtered):
        B, C, V, D = sino_filtered.shape
        X = self.geo["nVoxelX"]
        Y = self.geo["nVoxelY"]
        extent = self.geo["extent"]
        assert extent == 1, "This implementation currently assumes extent=1."

        sino_flat = sino_filtered.reshape(B, C, V * D)
        indices = self.geo["indices"].to(sino_filtered.device).view(X, Y, V * extent)
        indices_low = torch.floor(indices)
        indices_high = torch.ceil(indices)
        weight = torch.frac(indices)
        max_index = V * D - 1
        indices_low = torch.clamp(indices_low, 0, max_index).long()
        indices_high = torch.clamp(indices_high, 0, max_index).long()

        idx_low_flat = indices_low.flatten()
        idx_high_flat = indices_high.flatten()
        w_flat = weight.flatten().view(1, 1, -1)
        idx_low_expand = idx_low_flat.view(1, 1, -1).expand(B, C, -1)
        idx_high_expand = idx_high_flat.view(1, 1, -1).expand(B, C, -1)
        val_low = torch.gather(sino_flat, dim=2, index=idx_low_expand)
        val_high = torch.gather(sino_flat, dim=2, index=idx_high_expand)
        output = val_low * (1.0 - w_flat) + val_high * w_flat
        return output.view(B, C, X, Y, V * extent)

    def forward(self, sino):
        w1 = self.geo["w1"].to(sino.device).view(1, 1, 1, -1)
        x = sino * w1
        x = self._filter_sinogram(x.float())
        x = self._linear_backproject(x)
        w2 = self.geo["w2"].to(sino.device)
        x = x * w2
        x = torch.sum(x.float(), dim=-1) * np.pi / (self.geo["views"] * self.geo["extent"])
        return x
