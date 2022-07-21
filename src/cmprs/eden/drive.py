from cmprs.common import *
from cmprs.transform import Transform, FOut


class DriveQuantization(Transform):
    """
    This is EDEN specialized to 1-bit
    It should follow a randomized rotation (e.f., Hadamard transform)

    See 'DRIVE: One-bit Distributed Mean Estimation' (NeurIPS 2021) paper:
    https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html
    """

    def forward(self, x) -> FOut:
        # this is the unbiased version of DRIVE
        # an alternative version uses scale = l1(x) / d
        # for minimizing a single vector's MSE

        return FOut(torch.ge(x, 0, out=torch.empty_like(x)), sum_squares(x) / l1(x), bitrate=1)

    def backward(self, assignments, scale):
        return (assignments * 2 - 1) * scale


drive = DriveQuantization()
