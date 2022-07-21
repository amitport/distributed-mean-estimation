import torch

from cmprs.random_utils import hook_shared_prng
from cmprs.transform import Transform, FOut

torch.nn.init.orthogonal_(torch.empty(3, 5))


def _gen_random_orthogonal_matrix(d, device, dtype, prng):
    """Mostly Follow torch.nn.init.orthogonal_ for symmetric (semi) orthogonal matrices,
    which follows `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013)
    """

    Z = torch.empty([d, d], device=device, dtype=dtype).normal_(generator=prng.torch)
    Q, R = torch.linalg.qr(Z)

    return Q * torch.diag(R).sign()


class UniformRotation(Transform):
    def __init__(self, seed=None):
        self.prng = hook_shared_prng(self, seed)

    def get_rotation(self, x):
        return _gen_random_orthogonal_matrix(x.numel(), device=x.device, dtype=x.dtype, prng=self.prng)

    def forward(self, x) -> FOut:
        rotation = self.get_rotation(x)
        return FOut(rotation @ x)

    def backward(self, tx, rotation):
        rotation = self.get_rotation(tx)
        return tx @ rotation
