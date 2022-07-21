from typing import Optional

import numpy as np
import torch

from cmprs.random_utils import hook_shared_prng
from cmprs.rotations.hadamard import randomized_hadamard_transform, inverse_randomized_hadamard_transform
from cmprs.transform import Transform, FOut


def _kashin_padded_dim(dim, pad_threshold):
    if not dim & (dim - 1) == 0:
        padded_dim = int(2 ** (np.ceil(np.log2(dim))))
        if dim / padded_dim > pad_threshold:
            padded_dim = 2 * padded_dim
    else:
        padded_dim = 2 * dim

    return padded_dim


class KashinRepresentation(Transform):

    def __init__(self, eta=0.9, delta=1.0, pad_threshold=0.85, n_iters=3, seed: Optional[int] = None):
        self.prng = hook_shared_prng(self, seed)

        self.eta = eta
        self.delta = delta
        self.pad_threshold = pad_threshold
        self.n_iters = n_iters

    def forward(self, x):
        dim = x.numel()
        padded_dim = _kashin_padded_dim(dim, self.pad_threshold)

        kashin_coefficients = torch.zeros(padded_dim, device=x.device)
        padded_x = torch.zeros(padded_dim, device=x.device)

        M = torch.norm(x) / np.sqrt(self.delta * padded_dim)

        for i in range(self.n_iters):
            padded_x[:] = 0
            padded_x[:dim] = x
            padded_x = randomized_hadamard_transform(padded_x, self.prng.reset())

            b = padded_x
            b_hat = torch.clamp(b, min=-M, max=M)

            kashin_coefficients = kashin_coefficients + b_hat

            if i < self.n_iters - 1:
                b_hat = inverse_randomized_hadamard_transform(b_hat, self.prng.reset())
                x = x - b_hat[:dim]

                M = self.eta * M

        return FOut(kashin_coefficients, dim)

    def backward(self, qhvec, dim):
        return inverse_randomized_hadamard_transform(qhvec, self.prng)[:dim]
