import torch

from cmprs.transform import Transform, FOut
from cmprs.bitrate.recursive_elias_coding import RecursiveEliasCodingBitrate
from cmprs.bitrate.optimal import setBitrateFromMeasuredEntropy
from cmprs.bitrate.huffman_coding import huffmanCodingBitrate


class SetBitrate(Transform):
    def __init__(self, bitrate: float):
        self.bitrate = bitrate

    def forward(self, x):
        return FOut(tx=x, bitrate=self.bitrate)


class SetBitrateFromTensor(Transform):
    def forward(self, x):
        if x.dtype.is_floating_point:
            bitrate = torch.finfo(x.dtype).bits
        else:
            bitrate = torch.iinfo(x.dtype).bits
        return FOut(tx=x, bitrate=bitrate)


set_bitrate_from_tensor = SetBitrateFromTensor()


class VerifyIntBitrate(Transform):
    def __init__(self, bits):
        self.bits = bits

    def forward(self, x):
        x_int = x.to(torch.int64)
        int_vals = torch.max(x_int - torch.min(x_int)) + 1
        bitrate = torch.ceil(torch.log2(int_vals)).to(torch.int64)
        if self.bits < bitrate:
            raise RuntimeError(f'Input requires {bitrate} bits, but only {self.bits} are allowed '
                               f'(min={torch.min(x)}, max={torch.max(x)}, total value range={int_vals})')
        return FOut(tx=x, bitrate=self.bits)
