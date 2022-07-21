from typing import Optional

import torch

from cmprs.bitrate import SetBitrate, set_bitrate_from_tensor
from cmprs.random_utils import bernoulli_mask, hook_shared_prng
from cmprs.transform import Transform, FOut


def _mask_split(x, mask):
    x0 = torch.masked_select(x, torch.logical_not(mask))
    x1 = torch.masked_select(x, mask)
    return x0, x1


def _mask_combine(x0, x1, mask):
    x = torch.empty(mask.shape, dtype=x0.dtype, layout=x0.layout, device=x0.device)
    x.masked_scatter_(mask, x1)
    x.masked_scatter_(torch.logical_not(mask), x0)
    return x


class RandomPSplit(Transform):
    def __init__(self, p=0.5, seed: Optional[int] = None,
                 t0: Transform = set_bitrate_from_tensor,
                 t1: Transform = set_bitrate_from_tensor):
        self.p = p
        self.t0 = t0
        self.t1 = t1

        self.prng = hook_shared_prng(self, seed)

    def forward(self, x) -> FOut:
        original_shape = x.shape

        mask = bernoulli_mask(self.p, original_shape, x.device, self.prng)

        x0, x1 = _mask_split(x, mask)
        o0, o1 = self.t0.forward(x0), self.t1.forward(x1)

        if o0.bitrate is None or o1.bitrate is None:
            bitrate = None
        else:
            bitrate = self.p * o0.bitrate + (1 - self.p) * o1.bitrate

        return FOut(
            tx=(o0.tx, o1.tx),
            tcontext=(original_shape, (o0.tcontext, o1.tcontext)),
            bitrate=bitrate,
        )

    def backward(self, tx_s, context):
        tx0, tx1 = tx_s
        original_shape, (c0, c1) = context
        x0, x1 = self.t0.backward(tx0, c0), self.t1.backward(tx1, c1)

        mask = bernoulli_mask(self.p, original_shape, x0.device, self.prng)

        return _mask_combine(x0, x1, mask)


class SignAbsSplit(Transform):
    def __init__(self, t_abs: Transform = set_bitrate_from_tensor, t_sign: Transform = SetBitrate(1)):
        self.t_abs = t_abs
        self.t_sign = t_sign

    def forward(self, x) -> FOut:
        onebit_sign_x = torch.ge(x, 0, out=torch.empty_like(x))
        o_abs, o_sign = self.t_abs.forward(x.abs()), self.t_sign.forward(onebit_sign_x)

        if o_abs.bitrate is None or o_sign.bitrate is None:
            bitrate = None
        else:
            bitrate = o_abs.bitrate + o_sign.bitrate

        return FOut(
            tx=(o_abs.tx, o_sign.tx),
            tcontext=(o_abs.tcontext, o_sign.tcontext),
            bitrate=bitrate,
        )

    def backward(self, tx_s, context):
        abs_tx, sign_tx = tx_s
        abs_c, sign_c = context

        abs_x, onebit_sign_x = self.t_abs.backward(abs_tx, abs_c), self.t_sign.backward(sign_tx, sign_c)
        sign_x = onebit_sign_x * 2 - 1
        return abs_x * sign_x
