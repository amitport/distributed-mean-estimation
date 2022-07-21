from functools import cached_property, cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from cmprs.eden.quantization.quantizer import QData, Quantizer
import cmprs.config as config

ENTROPY_DELTAS_FILE = str(Path(__file__).with_name('entropy_deltas_mpf.npz'))
ENTROPY_DELTA_CENTROIDS_FILE = str(Path(__file__).with_name('entropy_delta_centroids_mpf.npz'))
ENTROPY_DELTA_EPS_FILE = str(Path(__file__).with_name('entropy_delta_eps_mpf.npz'))


class EntropyDeltaData(QData):
    @property
    def eps_file(self):
        return ENTROPY_DELTA_EPS_FILE

    @cached_property
    def clusters(self):
        deltas = np.load(ENTROPY_DELTAS_FILE, allow_pickle=True)
        deltas = {float(bits): np.float64(delta).item() for bits, delta in deltas.items()}

        all_pos_centroids = np.load(ENTROPY_DELTA_CENTROIDS_FILE, allow_pickle=True)
        all_pos_centroids = {float(bits): np.float64(c) for bits, c in all_pos_centroids.items()}

        all_centroids = {float(bits): np.concatenate([-np.flip(c), [0.], c]) for bits, c in all_pos_centroids.items()}

        return all_centroids, deltas

    def ep_for_bits(self, bits_low, bits_frac=None):
        if bits_frac is not None:
            bits = bits_low + bits_frac
        else:
            bits = bits_low
        if bits not in self.eps:
            raise ValueError(f'EntropyDeltaData missing precalculated {bits=}')
        return self.eps[bits]


@cache
def get_data_in_device(device):
    all_centroids, deltas = EntropyDeltaQuantizer.data.clusters
    all_centroids = {bits: torch.from_numpy(c).to(device) for bits, c in all_centroids.items()}
    return all_centroids, deltas


class EntropyDeltaQuantizer(Quantizer):
    data: ClassVar[EntropyDeltaData] = EntropyDeltaData()

    def __init__(self, bits: float):
        all_centroids, deltas = get_data_in_device(config.device)
        self.centroids = all_centroids[bits]
        self.n_values = len(self.centroids)
        self.delta = deltas[bits]

    def assign_clusters(self, x):
        """
        :param x: vector to be clustered
        :return: The cluster assignment vector (torch.int64).
                 The output includes negative values, and it is assumed
                 that a (near) optimal entropy coder follow.
        """

        x = torch.round(x / self.delta).to(torch.int64) + self.n_values // 2
        return torch.clamp(x, min=0, max=self.n_values - 1)

    def to_values(self, assignments):
        """
        :param assignments: cluster indices as returned by assign_clusters
        :return: each cluster index replaced with the cluster's centroid (torch.float64)
        """
        # theoretically, since the normal distribution is infinite we can get an assignment
        # value for which we did not calculate the centroid... in practice this does not happen
        # since we calculated many very un-probable centroids
        return torch.take(self.centroids, assignments)
