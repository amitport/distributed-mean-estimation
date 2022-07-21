import math
from typing import Literal, Type, Optional

import torch

from cmprs.bitrate import setBitrateFromMeasuredEntropy, SetBitrate, VerifyIntBitrate
from cmprs.common import l2, sqrt, sum_squares
from cmprs.eden.drive import drive
from cmprs.eden.quantization.entropy_delta import EntropyDeltaQuantizer
from cmprs.eden.quantization.opt import OptQuantizer
from cmprs.eden.quantization.quantizer import Quantizer
from cmprs.random_utils import hook_shared_prng, bernoulli_mask
from cmprs.reshape import flatten
from cmprs.rotations.hadamard import RandomizedHadamard, padToPowerOf2
from cmprs.sparsification import RandomP
from cmprs.split import RandomPSplit, _mask_split, _mask_combine
from cmprs.transform import Transform, SequenceTransform as Seq, FOut

QuantizationTypes = Literal['opt', 'entropy_delta']


class EdenExpectedScale(Transform):
    def __init__(self, Q: Type[Quantizer], bits_low, bits_frac=None):
        self.ep = Q.data.ep_for_bits(bits_low, bits_frac)

        # EDEN's scale = l2(x)^2 / <R(x), Q(R(x) * sqrt(d) / l2(x))>
        # Since R(x) * sqrt(d) / l2(x) ~ Normal(0, 1); we get in expectation:
        # scale = l2(x)^2 / ((l2(x) / sqrt(d)) * d * E(Z * Q(Z)))
        # = (l2(x)/sqrt(d)) / E(Z * Q(Z)) = std / ep

    def forward(self, x) -> FOut:
        d = x.numel()

        # post-rotation standard deviation:
        std = l2(x) / sqrt(d)

        return FOut(x / std, std / self.ep)

    def backward(self, tx, scale):
        return tx * scale


class EdenQ(Transform):
    def __init__(self, q: Quantizer):
        self.q = q

    def forward(self, x) -> FOut:
        return FOut(self.q.assign_clusters(x), x.dtype)

    def backward(self, assignments, orig_dtype):
        return self.q.to_values(assignments).to(orig_dtype)


class EdenFull(Transform):
    def __init__(self, q: Quantizer):
        self.q = q

    def forward(self, x) -> FOut:
        std = l2(x) / sqrt(x.numel())
        assignments = self.q.assign_clusters(x / std)
        values = self.q.to_values(assignments)
        x64 = x.to(torch.float64)
        scale = sum_squares(x64) / (values @ x64)
        return FOut(assignments, (scale, x.dtype))

    def backward(self, assignments, context):
        scale, orig_dtype = context
        return (self.q.to_values(assignments) * scale).to(orig_dtype)


class EdenOptFullSplit(Transform):
    def __init__(self, bits_low: int, bits_frac: float, seed: Optional[int] = None):
        self.q_s = OptQuantizer(bits_low), OptQuantizer(bits_low + 1)
        self.p = bits_frac

        self.prng = hook_shared_prng(self, seed)

    def forward(self, x) -> FOut:
        std = l2(x) / sqrt(x.numel())
        original_shape = x.shape

        mask = bernoulli_mask(self.p, original_shape, x.device, self.prng)
        assignments = [q.assign_clusters(_x) for q, _x in zip(self.q_s, _mask_split(x / std, mask))]
        values = _mask_combine(*[q.to_values(_a) for q, _a in zip(self.q_s, assignments)], mask)

        x64 = x.to(torch.float64)
        scale = sum_squares(x64) / (values @ x64)
        return FOut(assignments, (scale, x.dtype, original_shape))

    def backward(self, assignments, context):
        scale, orig_dtype, original_shape = context
        mask = bernoulli_mask(self.p, original_shape, assignments.device, self.prng)
        values = _mask_combine(*[q.to_values(_a) for q, _a in zip(self.q_s, assignments)], mask)
        return (values * scale).to(orig_dtype)


def eden(bits: int, quantization: QuantizationTypes = 'opt', use_expected_scale=True, split_seed: int = None,
         rotation_seed: int = None):
    rotation = [flatten, padToPowerOf2, RandomizedHadamard(rotation_seed)]

    if quantization == 'entropy_delta':
        if use_expected_scale:
            e = [EdenExpectedScale(EntropyDeltaQuantizer, bits), EdenQ(EntropyDeltaQuantizer(bits))]
        else:
            e = [EdenFull(EntropyDeltaQuantizer(bits))]
        return Seq([*rotation, *e, setBitrateFromMeasuredEntropy])

    if quantization != 'opt':
        raise ValueError(f'Unexpected {quantization=}')

    #  quantization == 'opt'

    if bits <= 1 + 1e-5:
        if use_expected_scale:
            e = [EdenExpectedScale(OptQuantizer, 1), EdenQ(OptQuantizer(1)), VerifyIntBitrate(1)]
        else:
            e = [drive]  # same as EdenFull(OptQuantizer(1))
        rest = Seq([*rotation, *e])
        if bits < 1 - 1e-5:
            # sparsify
            return RandomP(bits, split_seed, t=rest)
        else:
            # one-bit
            return rest
    else:
        # More than 1 bit
        bits_frac, bits_low = math.modf(bits)
        bits_low = int(bits_low)
        if bits_frac <= 1e-5:
            # integer bits
            if use_expected_scale:
                e = [EdenExpectedScale(OptQuantizer, bits_low), EdenQ(OptQuantizer(bits_low))]
            else:
                e = [EdenFull(OptQuantizer(bits_low))]
            return Seq([*rotation, *e, VerifyIntBitrate(bits_low)])
        else:
            # factional bits
            if use_expected_scale:
                return Seq([
                    *rotation,
                    EdenExpectedScale(OptQuantizer, bits_low, bits_frac),
                    RandomPSplit(bits_frac, split_seed,
                                 t0=Seq([EdenQ(OptQuantizer(bits_low)), VerifyIntBitrate(bits_low)]),
                                 t1=Seq([EdenQ(OptQuantizer(bits_low + 1)), VerifyIntBitrate(bits_low + 1)])),
                ])
            else:
                return Seq([*rotation, EdenOptFullSplit(bits_low, bits_frac, split_seed), SetBitrate(bits)])
