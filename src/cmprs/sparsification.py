from abc import ABC, abstractmethod
from typing import Optional

import torch

from cmprs.bitrate import set_bitrate_from_tensor
from cmprs.random_utils import hook_shared_prng, bernoulli_mask, random_k_mask
from cmprs.transform import Transform, FOut


class Sparsification(Transform, ABC):
    def __init__(self, seed: Optional[int] = None, t: Transform = set_bitrate_from_tensor):
        self.t = t
        self.prng = hook_shared_prng(self, seed)

    @abstractmethod
    def random_mask(self, shape, device):
        pass

    @abstractmethod
    def scale(self, x):
        pass

    def forward(self, x):
        original_shape = x.shape

        mask = self.random_mask(x.shape, x.device)
        scale = self.scale(x)

        tx = torch.masked_select(x, mask) * scale
        to = self.t.forward(tx)

        return FOut(
            tx=to.tx,
            tcontext=(original_shape, to.tcontext),
            bitrate=None if to.bitrate is None else to.bitrate / scale
        )

    def backward(self, sparse_tx, tcontext):
        original_shape, tc = tcontext
        sparse_x = self.t.backward(sparse_tx, tc)
        mask = self.random_mask(original_shape, sparse_x.device)

        x = torch.zeros(original_shape, dtype=sparse_x.dtype, device=sparse_x.device).masked_scatter_(mask, sparse_x)
        return x


class RandomP(Sparsification):
    """
    Random Sparsification given a parameter p remaining to determine the
    probability of a coordinate to keep
    """

    def __init__(self, p=0.5, seed: Optional[int] = None, t: Transform = set_bitrate_from_tensor):
        super().__init__(seed, t)
        self.p = p
        self._scale = 1 / self.p

    def random_mask(self, shape, device):
        return bernoulli_mask(self.p, shape, device, self.prng)

    def scale(self, x):
        return self._scale


class RandomK(Sparsification):
    """
    Random Sparsification given a parameter K to determine the number of
    coordinates to keep
    """

    def __init__(self, k, seed: Optional[int] = None, t: Transform = set_bitrate_from_tensor):
        super().__init__(seed, t)
        self.k = k

    def random_mask(self, shape, device):
        return random_k_mask(self.k, shape, device, self.prng)

    def scale(self, x):
        return x.numel() / self.k
