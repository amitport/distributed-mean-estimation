from functools import cache
import math

import torch


def l1(x): return torch.sum(torch.abs(x))


def sum_squares(x): return torch.sum(x ** 2)


def l2(x):
    # A bit slower but more accurate than torch.linalg.vector_norm
    return torch.sqrt(sum_squares(x))


@cache
def sqrt(n):
    # cached version for things like vectors' dimensions
    return math.sqrt(n)
