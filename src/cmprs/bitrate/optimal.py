import numpy as np
import torch

from cmprs.transform import Transform, FOut


class SetBitrateFromMeasuredEntropy(Transform):

    def forward(self, x):
        if x.is_floating_point():
            x_int = x.to(torch.int64)
        else:
            x_int = x
        x_int = x_int - torch.min(x_int)  # start at 0, avoid negative values
        freq = torch.bincount(x_int)
        freq = freq[freq.nonzero(as_tuple=True)]
        probs = freq / x.numel()

        # This is the entropy, which is the optimal bitrate:
        # https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem
        # there are coders such as arithmetic coding that come very close
        # (https://en.wikipedia.org/wiki/Arithmetic_coding)
        bitrate = torch.sum(-probs * torch.log2(probs)).item()

        return FOut(tx=x, bitrate=bitrate)


setBitrateFromMeasuredEntropy = SetBitrateFromMeasuredEntropy()