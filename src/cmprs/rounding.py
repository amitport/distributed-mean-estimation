from typing import Optional

import torch

from cmprs.random_utils import dither_like, hook_shared_prng
from cmprs.transform import Transform, FunctionalTransform, FOut


class StochasticRounding(Transform):
    def __init__(self, subtractive=False, seed: Optional[int] = None):
        self.subtractive = subtractive

        self.prng = hook_shared_prng(self, seed)

    def forward(self, x) -> FOut:
        """
            :param x: a *normalized* tensor in the [0, 2**bits-1] range
            :return: a tensor with the same dtype in {0, ... , 2**bits - 1},
                     and a random seed for the dither noise if 'subtractive' is True
        """

        return FOut(torch.round(x + dither_like(x, self.prng)))

    def backward(self, tx, _):
        if self.subtractive:
            tx = tx - dither_like(tx, self.prng)

        return tx


round_nearest_transform = FunctionalTransform(lambda x: FOut(torch.round(x)))
