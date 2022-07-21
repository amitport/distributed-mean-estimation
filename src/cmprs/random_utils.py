from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import torch

import cmprs.config as config
from cmprs.transform import Transform, FOut


@dataclass
class PRNG:
    torch: Any = None
    np: Any = None
    state: Any = None

    def reset(self):
        torch_seed, np_state = self.state

        self.np.bit_generator.state = np_state
        self.torch.manual_seed(torch_seed)
        return self



def bernoulli_mask(p: float, shape, device, prng: PRNG):
    """ returns random booleans """
    return torch.empty(shape, dtype=torch.bool, device=device).bernoulli_(p=p, generator=prng.torch)


def random_k_mask(k: int, shape, device, prng: PRNG):
    """ returns a random boolean tensor with exactly k True entries """
    d = np.prod(shape)
    mask = np.zeros(shape, dtype=bool)
    mask.put(prng.np.choice(d, size=k, replace=False), True)
    return torch.from_numpy(mask).to(device)


def dither_like(x, prng: PRNG):
    """ returns random reals in [-0.5, 0.5) """
    # having an open right bound is important for stochastic rounding
    # in order to avoid rounding the dither up when we get 2**bits-1 + 0.5
    return torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=prng.torch) - 0.5


def rademacher_like(x, prng: PRNG):
    """ returns a random vector in {-1, 1}**d """
    return torch.empty_like(x).bernoulli_(generator=prng.torch) * 2 - 1


UINT64_MAX = np.iinfo(np.uint64).max


def full_range_uint64(np_prng):
    return np_prng.integers(0, UINT64_MAX, dtype=np.uint64, endpoint=True).item()


def hook_shared_prng(transform: Transform, seed: Optional[int] = None):
    if seed is not None:
        seed = [config.root_seed, seed]
    ss = np.random.SeedSequence(seed)
    device = config.device

    np_prng = np.random.default_rng(ss)
    torch_prng = torch.Generator(device=device)

    prng = PRNG(torch=torch_prng, np=np_prng)

    t_forward = transform.forward
    t_backward = transform.backward

    def _forward(x):
        torch_seed = full_range_uint64(prng.np)

        np_state = prng.np.bit_generator.state
        prng.torch.manual_seed(torch_seed)

        prng.state = torch_seed, np_state

        return FOut(**asdict(t_forward(x)))

    def _backward(tx, context):
        prng.reset()

        return t_backward(tx, context)

    transform.forward = _forward
    transform.backward = _backward

    return prng
