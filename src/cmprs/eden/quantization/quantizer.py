from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

import numpy as np


class QData(ABC):
    @property
    @abstractmethod
    def eps_file(self):
        pass

    @property
    @abstractmethod
    def clusters(self):
        pass

    @abstractmethod
    def ep_for_bits(self, bits_low, bits_frac=None):
        pass

    @cached_property
    def eps(self):
        # expected inner product (for bias correction)
        expected_ips = np.load(self.eps_file, allow_pickle=True)
        expected_ips = {float(k): np.float64(v).item() for k, v in expected_ips.items()}
        return expected_ips


class Quantizer(ABC):
    data: ClassVar[QData]

    @abstractmethod
    def __init__(self, bits):
        pass

    @abstractmethod
    def assign_clusters(self, x):
        pass

    @abstractmethod
    def to_values(self, assignments):
        pass
