import torch.nn as nn
from .fbp import LInFBPFixedLinearFBPBatch


class FanBeamVVBPExtractor(nn.Module):
    """Extract fan-beam VVBP tensor using the same fixed FBP implementation as baseline.

    Input: sino [B, 1, views, nDetecU]
    Output: vvbp [B, 1, 512, 512, views]
    """
    def __init__(self, geo):
        super().__init__()
        self.geo = geo
        self.fbp = LInFBPFixedLinearFBPBatch(geo)

    def forward(self, sino):
        w1 = self.geo["w1"].to(sino.device).view(1, 1, 1, -1)
        x = sino * w1
        x = self.fbp._filter_sinogram(x.float())
        x = self.fbp._linear_backproject(x)
        w2 = self.geo["w2"].to(sino.device)
        x = x * w2
        return x
