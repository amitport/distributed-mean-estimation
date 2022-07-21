from functools import cache, cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

import cmprs.config as config
from cmprs.eden.quantization.quantizer import QData, Quantizer

OPT_CENTROIDS_FILE = str(Path(__file__).with_name('opt_centroids_mpf.npz'))
OPT_EPS_FILE = str(Path(__file__).with_name('opt_eps_mpf.npz'))


class OptData(QData):
    @property
    def eps_file(self):
        return OPT_EPS_FILE

    @cached_property
    def clusters(self):
        half_centroids = np.load(OPT_CENTROIDS_FILE, allow_pickle=True)
        half_centroids = {int(k): np.asarray(v, dtype=np.float64) for k, v in half_centroids.items()}

        half_boundaries = {bits: (c[:-1] + c[1:]) / 2 for bits, c in half_centroids.items()}
        centroids = {i: np.concatenate([-np.flip(c), c]) for i, c in half_centroids.items()}
        boundaries = {i: np.concatenate([-np.flip(b), [0.], b]) for i, b in half_boundaries.items()}
        return centroids, boundaries

    def ep_for_bits(self, bits_low, bits_frac=None):
        if bits_frac is None:
            return self.eps[bits_low]
        else:
            return bits_frac * self.eps[bits_low] + (1 - bits_frac) * self.eps[bits_low + 1]


@cache
def get_opt_in_device(device):
    centroids, boundaries = OptQuantizer.data.clusters
    centroids = {i: torch.from_numpy(c).to(device) for i, c in centroids.items()}
    boundaries = {i: torch.from_numpy(b).to(device) for i, b in boundaries.items()}
    return centroids, boundaries


class OptQuantizer(Quantizer):
    data: ClassVar[OptData] = OptData()

    def __init__(self, bits: int):
        centroids, boundaries = get_opt_in_device(config.device)
        self.centroids = centroids[bits]
        self.boundaries = boundaries[bits]

    def assign_clusters(self, x):
        """
        :param x: vector to be clustered
        :return: The cluster assignment vector (cluster indices) (torch.int64).
        """
        return torch.bucketize(x, self.boundaries)

    def to_values(self, assignments):
        """
        :param assignments: cluster indices as returned by assign_clusters
        :return: each cluster index replaced with the cluster's centroid (torch.float64)
        """
        return torch.take(self.centroids, assignments)
