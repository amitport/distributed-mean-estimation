import torch

from cmprs.transform import Transform, FOut


class TernGradClip(Transform):
    def forward(self, x):
        th = 2.5 * torch.std(x)
        clipped_vec = torch.clamp(x, -th, th)

        return FOut(clipped_vec)


ternGradClip = TernGradClip()
