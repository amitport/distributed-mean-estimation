from math import floor, log2
from typing import Optional

import torch
import torch.nn.functional as F

from cmprs.common import sqrt
from cmprs.random_utils import rademacher_like, hook_shared_prng
from cmprs.transform import Transform, FOut


def hadamard_transform(vec):
    """fast Walsh–Hadamard transform

    hadamard transform is not very numerically stable by nature (lots of subtractions)
    should try and use with float64 when possible

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    d = vec.numel()
    original_shape = vec.shape
    h = 2
    while h <= d:
        hf = h // 2
        vec = vec.view(d // h, h)

        # For an in=place version we could use
        #  vec[:, :hf] += vec[:, hf:]
        #  vec[:, hf:] *= -2
        #  vec[:, hf:] += vec[:, :hf]
        # Generally, the in-place version is not recommended
        # since it is generally *not* faster, and the in-place
        # side effect can easily introduce bugs.
        # *It is appropriate when we otherwise run out of memory*.

        # TODO add batch support with
        # half_1, half_2 = batch[:, :, :hf], batch[:, :, hf:]
        half_1, half_2 = vec[:, :hf], vec[:, hf:]

        vec = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (vec / sqrt(d)).view(*original_shape)


def randomized_hadamard_transform(x, prng):
    d = rademacher_like(x, prng)

    return hadamard_transform(x * d)


def inverse_randomized_hadamard_transform(tx, prng):
    d = rademacher_like(tx, prng)

    return hadamard_transform(tx) * d


class RandomizedHadamard(Transform):
    """
      Assumes that the input is a vector and a power of 2
    """

    def __init__(self, seed: Optional[int] = None):
        self.prng = hook_shared_prng(self, seed)

    def forward(self, x) -> FOut:
        return FOut(randomized_hadamard_transform(x, self.prng))

    def backward(self, tx, seed):
        return inverse_randomized_hadamard_transform(tx, self.prng)


class Hadamard(Transform):
    """
      Assumes that the input is a vector and a power of 2
    """

    def forward(self, x) -> FOut:
        return FOut(hadamard_transform(x))

    def backward(self, tx, seed):
        return hadamard_transform(tx)


def next_power_of_2(n):
    return 2 ** (floor(log2(n)) + 1)


def is_a_power_of_2(n):
    return n & (n - 1) == 0


class PadToPowerOf2(Transform):
    def forward(self, x) -> FOut:
        """_
        :param x: assumes vec is 1d
        :return: x padded with zero until the next power-of-2
        """
        d = x.numel()
        # pad to the nearest power of 2 if needed
        if is_a_power_of_2(d):
            return FOut(x, d)
        else:
            return FOut(F.pad(x, (0, next_power_of_2(d) - d)), d)

    def backward(self, tx, original_dim):
        return tx[:original_dim]


padToPowerOf2 = PadToPowerOf2()
