import os

import numpy as np
import torch
from mpmath import mp
from tqdm import tqdm

from eden.quantization.cluster_distribution import normal_cluster_dist as norm_c
from eden.quantization.entropy_delta import ENTROPY_DELTAS_FILE, ENTROPY_DELTA_CENTROIDS_FILE, ENTROPY_DELTA_EPS_FILE
from eden.quantization.quantizer import Quantizer

DPS = 150


def mad_ratio(delta):
    max_c = 0

    centroid_mad = round_center_mad = 0  # conceptually mass at (-0.5*d, 0.5*d) * 0 (centroid = 0)
    # add the first non-central interval
    n = 1
    interval = (delta * (n - 0.5), delta * (n + 0.5))
    p = norm_c.mass(*interval)
    # In practice very low p stopping condition should be enough to
    # avoid any chance to draw a number for which we didn't calculate the centroid:
    while p > 1e-17:
        centroid = norm_c.centroid(*interval)
        if centroid < max_c:
            # numeric instability...
            break
        else:
            max_c = centroid

        centroid_mad += p * centroid
        round_center_mad += p * (delta * n)
        # conceptually both values above should be multiplied by 2 since we're only calculating positive values
        # but there is no need to do this since we're interested in the ratio

        n += 1
        interval = (delta * (n - 0.5), delta * (n + 0.5))
        p = norm_c.mass(*interval)

    return centroid_mad / round_center_mad


def calc_centroids(delta):
    centroids = []
    max_c = 0

    # add the first non-central interval
    n = 1
    interval = (delta * (n - 0.5), delta * (n + 0.5))
    p = norm_c.mass(*interval)
    # In practice very low p stopping condition should be enough to
    # avoid any chance to draw a number for which we didn't calculate the centroid:
    while p > 1e-17:
        centroid = norm_c.centroid(*interval)
        if centroid < max_c:
            # numeric instability...
            break
        else:
            max_c = centroid
        centroids.append(centroid)
        # conceptually both values above should be multiplied by 2 since we're only calculating positive values
        # but there is no need to do this since we're interested in the ratio

        n += 1
        interval = (delta * (n - 0.5), delta * (n + 0.5))
        p = norm_c.mass(*interval)

    centroids = np.float64(centroids)
    centroids = np.concatenate([-np.flip(centroids), [0.], centroids])
    return torch.from_numpy(centroids)


class DeltaCentroidQuantizer(Quantizer):

    def __init__(self, sqrt_2_raw_moment, has_zero_cluster, delta, stop_at_prob=0):
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
