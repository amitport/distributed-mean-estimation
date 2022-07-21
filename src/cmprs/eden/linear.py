from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import List

import torch
from scipy.stats import norm

import cmprs.config as config
from cmprs.transform import Transform, FOut


def calc_expected_round_product(has_zero_cluster, delta, stop_at_prob=0):
    # for X ~ N(0, (1 / delta) ** 2)
    # when has_zero_cluster=True --> calculate E[X*Round(X)]
    # when has_zero_cluster=False --> calculate E[X * (Round(X + ½ sign(X)) - ½ sign(X))]
    p = float('inf')
    product = 0
    if has_zero_cluster:
        n = 1 / 2
        # we will safely skip the 0..0.5 interval since it rounds to zero
    else:
        n = 0
    while p > stop_at_prob:
        l, r = n, (n + 1)
        e = norm.expect(lb=l, ub=r,
                        scale=1 / delta)  # this is the same as centroid * probability_mass (since centroid == e/probability_mass)
        product += e * (n + 1 / 2)
        n += 1
        p = norm.sf(r, scale=1 / delta)  # survival, which is more numerically stable 1 - cdf
    return 2 * product  # multiply by 2 since we only covered positive values


def calc_expected_round_product2(has_zero_cluster, delta, stop_at_prob=0):
    product = 0
    if has_zero_cluster:
        n = 1 / 2
        # we can safely skip the 0..0.5 where centroid and round center agree
    else:
        n = 0

    l, r = n, (n + 1)
    p = (norm.cdf(r, scale=1 / delta) - norm.cdf(l, scale=1 / delta))
    while p > stop_at_prob:
        centroid = norm.expect(lb=l, ub=r,
                               scale=1 / delta) / p  # this is the same as centroid * probability_mass (since centroid === e/probability_mass)
        product += p * centroid ** 2
        n += 1
        l, r = n, (n + 1)
        p = (norm.cdf(r, scale=1 / delta) - norm.cdf(l, scale=1 / delta))
    return 2 * product


def calc_expected_normal_bias_correction(has_zero_cluster, delta, stop_at_prob=0):
    if has_zero_cluster:
        n = 1 / 2
        # we can safely skip the 0..0.5 where centroid and round center agree
    else:
        n = 0
    centroid_mad = round_center_mad = 0

    l, r = n, (n + 1)
    p = (norm.cdf(r, scale=1 / delta) - norm.cdf(l, scale=1 / delta))
    while p > stop_at_prob:
        centroid_mad += norm.expect(lb=l, ub=r,
                                    scale=1 / delta)  # this is the same as centroid * probability_mass (since centroid === e/probability_mass)
        round_center_mad += p * (n + 1 / 2)
        n += 1
        l, r = n, (n + 1)
        p = (norm.cdf(r, scale=1 / delta) - norm.cdf(l, scale=1 / delta))
    return centroid_mad / round_center_mad


def calc_expected_deviation_ratio(has_zero_interval, delta, stop_at_interval_mass=1e-15, num_of_positive_intervals=float('inf')):
    intervals = pos_normal_intervals(has_zero_interval, delta, stop_at_interval_mass, num_of_positive_intervals)[0]

    c_deviation, lc_deviation = zip(*[(interval.mass * interval.c, interval.mass * interval.lc) for interval in intervals])

    return sum(c_deviation) / sum(lc_deviation)


def calc_expected_products(has_zero_interval, delta, stop_at_interval_mass=1e-15, num_of_positive_intervals=float('inf')):
    intervals = pos_normal_intervals(has_zero_interval, delta, stop_at_interval_mass, num_of_positive_intervals)[0]

    c_product, lc_product = zip(*[(interval.mass * interval.c ** 2, interval.mass * interval.c * interval.lc) for interval in intervals])

    return sum(c_product), sum(lc_product)


@dataclass
class Interval:
    lb: float  # lower bound
    ub: float  # upper bound
    c: float  # centroid (conditional expectation at (lb, ub))
    lc: float  # linear center (lb + delta / 2)
    mass: float  # cdf(ub) - cdf(lb)


@dataclass
class Intervals:
    intervals: List[Interval]
    clip_prob: float  # probability of falling outside [-(lb+delta), (lb+delta)] for max lower bound
    # for X ~ N(0, (1 / delta) ** 2)
    expected_lc_product: float  # E[X*Round(X)] or E[X * (Round(X + ½ sign(X)) - ½ sign(X))] depending on has_zero_interval
    expected_c_product: float   # E[X*Q(X)] where Q is the centroid of the interval containing X:
                                #    for has_zero_interval = True
                                #           (-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), ...
                                #           or
                                #    for has_zero_interval = False
                                #           (Round(X + ½ sign(X)) - ½ sign(X) - ½, Round(X + ½ sign(X)) - ½ sign(X) + ½)

@cache
def pos_normal_intervals(has_zero_interval, delta, stop_at_interval_mass=1e-15, num_of_positive_intervals=float('inf')):
    intervals: List[Interval] = []
    scale = 1 / delta
    if has_zero_interval:
        n = 1 / 2
    else:
        n = 0

    last_interval = False
    while not last_interval:
        lb, ub = n, n + 1
        ub_sf = norm.sf(ub, scale=scale)
        if n + 1 >= num_of_positive_intervals or ub_sf <= stop_at_interval_mass:
            # there will be no right interval point
            # effectively we clip values at upper bound
            clip_prob = 2 * ub_sf
            ub = float('inf')
            ub_sf = 0
            last_interval = True
        mass = norm.sf(lb, scale=scale) - ub_sf

        c = norm.expect(lb=lb, ub=ub, scale=scale, conditional=True)
        lc = lb + 1 / 2
        intervals.append(Interval(lb, ub, c, lc, mass))

        n += 1

    return intervals, clip_prob


def calc_centroids(has_zero_interval, delta,
                   stop_at_interval_mass=0, num_of_positive_intervals=float('inf')):
    max_c = 0
    centroids = []
    scale = 1 / delta
    if has_zero_interval:
        n = 1 / 2
    else:
        n = 0

    lb = n
    if n + 1 >= num_of_positive_intervals:
        # there is no right interval point
        ub = float('inf')
    else:
        ub = n + 1
    cdf_lb, cdf_ub = norm.cdf(lb, scale=scale), norm.cdf(ub, scale=scale)
    interval_mass = cdf_ub - cdf_lb

    while interval_mass > stop_at_interval_mass:
        centroid = norm.expect(lb=lb, ub=ub, scale=scale, conditional=True)
        if centroid <= max_c:
            # numeric instability...
            break
        else:
            max_c = centroid
        centroids.append(delta * centroid)

        n += 1

        lb = n
        if n + 1 >= num_of_positive_intervals:
            # there is no right interval point
            ub = float('inf')
        else:
            ub = n + 1
        cdf_lb, cdf_ub = cdf_ub, norm.cdf(ub, scale=scale)
        interval_mass = cdf_ub - cdf_lb

    negative_centroids = [-c for c in reversed(centroids)]
    if has_zero_interval:
        return [*negative_centroids, 0, *centroids]
    else:
        return [*negative_centroids, *centroids]


@cache
def calc_correction(has_zero_cluster, delta, stop_at_prob=0):
    # We divide by the expected product and multiply
    # by the constant normal bias correction that exists
    # from rounding to equal intervals instead of centroids
    # Also note that we divide by delta and not delta**2
    # since multiplying by delta is needed for final rescaling
    a = calc_expected_normal_bias_correction(has_zero_cluster, delta, stop_at_prob)
    b = calc_expected_round_product2(has_zero_cluster, delta, stop_at_prob)
    return a / (b * delta)


def round_to_half_int(x):
    """round the input to the nearest ½𝓩"""
    half_sign = torch.sign(x) / 2
    # round towards nearest ½𝓩
    # even integers go towards zero, uneven integers go away from zero
    x = torch.round(x + half_sign) - half_sign
    # we randomly round zeros towards {-0.5, 0.5}
    zeros_mask = x.eq(0)
    n_zeros = torch.count_nonzero(zeros_mask)
    dither = torch.round(torch.rand(n_zeros)) - 0.5  # random values in {-0.5, 0.5}
    return x.masked_scatter_(zeros_mask, dither)


class LinearEden(Transform):
    def __init__(self, sqrt_2_raw_moment, has_zero_cluster, delta, stop_at_prob=0):
        self.has_zero_cluster = has_zero_cluster
        self.delta = delta

        # EDEN's scale = l2(x)^2 / <R(x), Q(R(x) * sqrt(d) / l2(x))>
        #   Q(R(x) * sqrt(d) / l2(x)) -> Round(R(x) * sqrt(d) / l2(x) / delta) * delta
        #   Since (R(x) * sqrt(d) / l2(x) / delta) ~ N(0, (1 / delta) ** 2)
        #   we have <R(x), Q(R(x) * sqrt(d) / l2(x))> = (l2(x) / sqrt(d)) * delta * d * E[X * Q(X)] * delta
        #   => l2(x)^2 / <R(x), Q(R(x) * sqrt(d) / l2(x)) = (l2(x)/sqrt(d))/ (delta**2 * E[X * Q(X)])
        #                                                 = std / (delta**2 * ep)

        # for linearity we also need an estimate of input coordinate std
        # since post-Hadamard std is l2/sqrt(d) and E(l2) = sqrt(d * E(x**2)) for coordinate x
        # the expected post-Hadamard std is sqrt(E(x**2)) (this is sqrt of the 2-raw-moment)
        # (if the input vector coordinates have zero mean this is their std)
        self.std = sqrt_2_raw_moment  # should be the pre-hadamard coordinates' sqrt(E(x**2))

        # this is std * normal_corr / (ep * delta)
        self.constant_scale = self.std * calc_correction(has_zero_cluster, delta, stop_at_prob)

    def forward(self, x) -> FOut:
        tx = x / self.std / self.delta
        if self.has_zero_cluster:
            tx = torch.round(tx)
        else:
            tx = round_to_half_int(tx)
        # (instead of multiplying tx by delta here, the constant scale is divided by delta)

        return FOut(tx, self.constant_scale)

    def backward(self, tx, scale):
        return tx * scale


class DeltaCentroidQuantizer:
    def __init__(self, has_zero_cluster, delta, stop_at_prob=0):
        centroids = calc_centroids(has_zero_cluster, delta, stop_at_prob)
        self.centroids = torch.tensor(centroids, dtype=torch.float64).to(config.device)
        self.n_values = len(self.centroids)

        self.has_zero_cluster = has_zero_cluster
        self.delta = delta

    def assign_clusters(self, x):
        """
        :param x: vector to be clustered
        :return: The cluster assignment vector (torch.int64).
                 The output includes negative values, and it is assumed
                 that a (near) optimal entropy coder follow.
        """

        tx = x / self.delta
        if self.has_zero_cluster:
            tx = torch.round(tx)
        else:
            tx = round_to_half_int(tx)

        x = (tx + self.n_values // 2 + (-0.5 if not self.has_zero_cluster else 0)).to(torch.int64)
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
