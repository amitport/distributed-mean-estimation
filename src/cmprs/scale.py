import torch

from cmprs.transform import Transform, FOut


class MinMaxNormalization(Transform):
    """
    Maps a real vector into [0..1] range: (x - min(x)) / (max(x) - min(x))
    """

    def forward(self, x) -> FOut:
        minimum = x.min()
        maximum = x.max()

        return FOut(torch.nan_to_num((x - minimum) / (maximum - minimum)), (minimum, maximum))

    def backward(self, tx, context):
        minimum, maximum = context
        return tx * (maximum - minimum) + minimum


minmax_normalization = MinMaxNormalization()


class UnitNormalization(Transform):
    """
    Normalizes a real vector by dividing it by its norm (the resulting vector's norm is 1)
    """

    def __init__(self, norm_ord):
        self.ord = norm_ord

    def forward(self, x) -> FOut:
        x_norm = torch.linalg.vector_norm(x, ord=self.ord)

        return FOut(torch.nan_to_num(x / x_norm), x_norm)

    def backward(self, tx, x_norm):
        return tx * x_norm


l1_normalization = UnitNormalization(norm_ord=1)
l2_normalization = UnitNormalization(norm_ord=2)
max_normalization = UnitNormalization(norm_ord=float('inf'))


class ConstScale(Transform):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, x) -> FOut:
        return FOut(x * self.scale)

    def backward(self, tx, _):
        return tx / self.scale


class ConstOffset(Transform):
    def __init__(self, offset):
        self.offset = offset

    def forward(self, x) -> FOut:
        return FOut(x + self.offset)

    def backward(self, tx, _):
        return tx - self.offset
