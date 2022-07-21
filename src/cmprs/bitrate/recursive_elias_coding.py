from functools import cache
from math import floor, log2

import torch

from cmprs.transform import Transform, FOut


# recursive_elias a.k.a.elias omega coding

@cache
def recursive_elias_code_len(n):
    if n < 1:
        raise ValueError('elias omega code only accept positive integers')
    if n == 1:
        return 1
    num_bits = floor(log2(n)) + 1
    return num_bits + recursive_elias_code_len(num_bits - 1)


@cache
def elias_n_2_code_len(bits, device):
    return torch.tensor([recursive_elias_code_len(n) for n in range(1, 2 ** bits + 1)], dtype=torch.float32,
                        device=device)


def elias_bitrate(x, bits):
    #  assumes x is in [0, 2**bits - 1]
    n_2_code_len = elias_n_2_code_len(bits, x.device)
    if x.is_floating_point():
        x = torch.round(x)
    return torch.mean(torch.take(n_2_code_len, x.to(torch.int64)))


class RecursiveEliasCodingBitrate(Transform):
    def __init__(self, bits):
        self.bits = bits

    def forward(self, x) -> FOut:
        bitrate = elias_bitrate(x, self.bits)

        return FOut(tx=x, bitrate=bitrate)
