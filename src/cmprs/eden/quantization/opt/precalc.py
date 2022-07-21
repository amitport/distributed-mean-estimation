import os

import numpy as np
from matplotlib import pyplot as plt
from mpmath import mp

from cmprs.eden.quantization.cluster_distribution import normal_cluster_dist as norm_c
from cmprs.eden.quantization.opt import OPT_CENTROIDS_FILE, OPT_EPS_FILE
from cmprs.eden.quantization.opt.lloyd_max_1d_symmetric import LloydMax1DSymmetric

BITS = 8
DPS = 150


def precalc_opt_centroids(plot_error=True, steps=500, early_stopping=True, eq_rounds_to_stop=100):
    lloyd_max = LloydMax1DSymmetric()

    if os.path.isfile(OPT_CENTROIDS_FILE):
        all_pos_centers = {k: v for k, v in np.load(OPT_CENTROIDS_FILE, allow_pickle=True).items()}
    else:
        all_pos_centers = {str(b): np.arange(1, 2 ** (b - 1) + 1).astype(np.float64) / 2 ** (BITS - 1) for b in
                           range(1, BITS + 1)}
    # note that the 1-bit solution can also be
    # calculated directly with arbitrary precision via:
    # `mp.sqrt(2) / mp.sqrt(mp.pi)`
    for bits in range(1, BITS + 1):
        print(f'Start {bits=}')

        centers, errors = lloyd_max.run(all_pos_centers[str(bits)], steps, early_stopping, eq_rounds_to_stop)

        if plot_error:
            plt.plot(errors)
            plt.ylabel('distortion')
            plt.xlabel('steps')
            plt.title(f'{bits=}')
            plt.show()
        all_pos_centers[str(bits)] = centers
    np.savez(OPT_CENTROIDS_FILE, **all_pos_centers)


def precalc_expected_normal_q_product():
    """
    This calculates, for every bit, the expected value of multiplying a Z ~ N(0, 1) by its allocated quantization value
    i.e., E(Z * Q_b(Z))
    """
    if os.path.isfile(OPT_CENTROIDS_FILE):
        all_pos_centers = np.load(OPT_CENTROIDS_FILE, allow_pickle=True)
    else:
        raise ValueError('precalc_expected_inner_product requires valid centroids')

    all_eips = {}
    for bits in range(1, BITS + 1):
        print(f'Start {bits=}')

        centroids = all_pos_centers[str(bits)]
        # note this assumes that we have true centroids
        # which lie in the middle between two boundaries
        # (which is true *after* LLoyd-Max)
        boundaries = LloydMax1DSymmetric.get_boundaries(centroids)

        centroids_mass = [norm_c.mass(left, right) for left, right
                          in zip([0, *boundaries], [*boundaries, mp.inf])]
        # this assumes that our quantization assign the centroid to each cluster (which is true)
        # multiplying by 2 since we're only going through the positive clusters of the normal distribution
        eip = 2 * sum([m * c ** 2 for c, m in zip(centroids, centroids_mass)])

        all_eips[str(bits)] = eip

    np.savez(OPT_EPS_FILE, **all_eips)


def print_opt_mse(all_pos_centers):
    for bits, centers in all_pos_centers.items():
        centers = all_pos_centers[bits]
        boundaries = LloydMax1DSymmetric.get_boundaries(centers)
        print(f'{bits} bits normal mse = {norm_c.symmetric_full_mse(centers, boundaries)}')


def print_opt_as_literal(all_pos_centers):
    all_pos_centers = {int(k): v.astype(np.float64) for k, v in all_pos_centers.items()}

    print('opt_hn_centroids = {')
    for k, v in all_pos_centers.items():
        print(f'    {k}: [{", ".join([str(_) for _ in v])}],')
    print('}')


def print_eps_as_literal(all_eps):
    all_eps = {int(k): v.astype(np.float64) for k, v in all_eps.items()}
    print('eps = {' + ', '.join([f'{k}: {v}' for k, v in all_eps.items()]) + '}')


if __name__ == '__main__':
    """ uncomment below as needed for pre-calculation"""
    # from mpmath import workdps
    # with workdps(DPS):
    #   precalc_opt_centroids()
    #   precalc_expected_normal_q_product()

    # print_opt_mse(np.load(OPT_CENTROIDS_FILE, allow_pickle=True))
    # print_opt_as_literal(np.load(OPT_CENTROIDS_FILE, allow_pickle=True))
    # print_eps_as_literal(np.load(OPT_EPS_FILE, allow_pickle=True))
    print('Done')
