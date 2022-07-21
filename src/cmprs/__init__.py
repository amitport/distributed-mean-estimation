import math
from typing import Literal

from cmprs.clipping import ternGradClip
from cmprs.eden import eden
from cmprs.reshape import flatten
from cmprs.bitrate import VerifyIntBitrate, RecursiveEliasCodingBitrate
from cmprs.rotations.hadamard import padToPowerOf2, RandomizedHadamard
from cmprs.rounding import StochasticRounding, round_nearest_transform
from cmprs.scale import minmax_normalization, ConstScale, UnitNormalization
from cmprs.sparsification import RandomP
from cmprs.split import RandomPSplit, SignAbsSplit
from cmprs.transform import SequenceTransform as Seq


def _with_factional_bits_support(quantization_fn, bits, sparsity_seed=None):
    """

    :param quantization_fn: a quantization that only support natural numbers, expects one parameter: natural bits
    :param bits: a float representing possible fractional bit rates
    :param sparsity_seed:
    :return:
    """
    if bits < 1:
        # sparsify to support sub 1 bit rates
        return RandomP(bits, sparsity_seed, t=Seq([quantization_fn(1), VerifyIntBitrate(1)]))
    else:
        bits_frac, bits_low = math.modf(bits)
        bits_low = int(bits_low)
        if bits_frac <= 1e-5:  # it's not worth the effort if frac less than 1 in 100_000
            # use standard quantization
            return quantization_fn(bits_low)
        else:
            # random split to support fractional bit rates
            # the fraction determines how many will be assigned to bits_low
            return RandomPSplit(bits_frac, sparsity_seed,
                                t0=Seq([quantization_fn(bits_low), VerifyIntBitrate(bits_low)]),
                                t1=Seq([quantization_fn(bits_low + 1), VerifyIntBitrate(bits_low + 1)]))


RoundingType = Literal['randomized', 'subtractive', 'nearest']


def rounding(rounding_type: RoundingType, bits, normalization=minmax_normalization, rounding_seed=None,
             sparsity_seed=None, rotation=False, rotation_seed=None):
    if rounding_type == 'randomized':
        _rounding = StochasticRounding(subtractive=False, seed=rounding_seed)
    elif rounding_type == 'subtractive':
        _rounding = StochasticRounding(subtractive=True, seed=rounding_seed)
    elif rounding_type == 'nearest':
        _rounding = round_nearest_transform
    else:
        raise ValueError(f'Invalid {rounding_type=}')

    def _quantization_fn(b):
        return Seq([normalization, ConstScale(2 ** b - 1), _rounding])

    r = _with_factional_bits_support(_quantization_fn, bits, sparsity_seed)

    if rotation:
        r = Seq([flatten, padToPowerOf2, RandomizedHadamard(rotation_seed), r])

    return r


def qsgd(bits, norm_ord=2, rounding_seed=None, sparsity_seed=None):
    if bits <= 1 + 1e-5:
        raise ValueError('QSGD require more than 1 bit')

    r = Seq([rounding(rounding_type='randomized',
                      bits=bits - 1,  # one bit will be used to the sign
                      normalization=UnitNormalization(norm_ord=norm_ord),
                      rounding_seed=rounding_seed,
                      sparsity_seed=sparsity_seed), RecursiveEliasCodingBitrate(bits - 1)])

    return SignAbsSplit(t_abs=r)


def terngrad(rounding_seed=None):
    return Seq([
        ternGradClip,
        UnitNormalization(norm_ord=float('inf')),
        StochasticRounding(subtractive=False, seed=rounding_seed),
        VerifyIntBitrate(2)
    ])
