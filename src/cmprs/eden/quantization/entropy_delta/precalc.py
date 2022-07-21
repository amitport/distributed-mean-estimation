import os

import numpy as np
from mpmath import mp
from tqdm import tqdm

from cmprs.eden.quantization.cluster_distribution import normal_cluster_dist as norm_c
from cmprs.eden.quantization.entropy_delta import ENTROPY_DELTAS_FILE, ENTROPY_DELTA_CENTROIDS_FILE, ENTROPY_DELTA_EPS_FILE

BITS = [i / 10 for i in range(1, 10)] + [i for i in range(1, 9)]
DPS = 150


def entropy_from_delta(delta):
    # add the central interval
    p = norm_c.mass(delta * -0.5, delta * 0.5)
    entropy = -p * mp.log(p, b=2)

    # go over the positive interval and multiply each by 2
    n = 1
    while True:
        p = norm_c.mass(delta * (n - 0.5), delta * (n + 0.5))
        if p == 0:
            break
        entropy += 2 * (-p * mp.log(p, b=2))
        n += 1

    return entropy


def find_delta(bits, printer=print):
    delta_high = 10
    delta_low = 0

    delta_mid = mp.mpf(delta_high + delta_low) / 2
    entropy_bits = entropy_from_delta(delta_mid)

    bit_dist = abs(entropy_bits - bits)
    while bit_dist > 1e-50:

        if entropy_bits < bits:
            delta_high = delta_mid
        elif entropy_bits > bits:
            delta_low = delta_mid

        delta_mid = (delta_high + delta_low) / 2
        entropy_bits = entropy_from_delta(delta_mid)
        bit_dist = abs(entropy_bits - bits)
        printer(f'Entropy Delta , {bits=}, shared digits={int(-mp.log10(bit_dist))}')

    return delta_mid


def precalc_entropy_delta():
    if os.path.isfile(ENTROPY_DELTAS_FILE):
        # make sure we don't accidentally override previous values
        all_deltas = {k: v for k, v in np.load(ENTROPY_DELTAS_FILE, allow_pickle=True).items()}
    else:
        all_deltas = {}
    for b in (pbar := tqdm(BITS)):
        all_deltas[str(b)] = find_delta(b, lambda _: pbar.set_description(_))

    np.savez(ENTROPY_DELTAS_FILE, **all_deltas)


def precalc_entropy_delta_centroids():
    if os.path.isfile(ENTROPY_DELTAS_FILE):
        all_deltas = {k: v for k, v in np.load(ENTROPY_DELTAS_FILE, allow_pickle=True).items()}
    else:
        raise ValueError('precalc_entropy_delta_centroids requires valid deltas')

    all_centroids = {}

    for bits, d in (pbar := tqdm(all_deltas.items())):
        max_c = 0
        c = []
        # add the first non-central interval
        p = norm_c.mass(d * 0.5, d * 1.5)
        n = 1
        # In practice very low p stopping condition should be enough to
        # avoid any chance to draw a number for which we didn't calculate the centroid:
        while p > 1e-17:
            pbar.set_description(f'{bits=} #centroid={n}')
            centroid = norm_c.centroid(d * (n - 0.5), d * (n + 0.5))
            if centroid < max_c:
                # numeric instability...
                break
            else:
                max_c = centroid
            c.append(centroid)
            n += 1
            p = norm_c.mass(d * (n - 0.5), d * (n + 0.5))

        all_centroids[bits] = c

    np.savez(ENTROPY_DELTA_CENTROIDS_FILE, **all_centroids)


def precalc_expected_entropy_delta_q_product():
    """
    This calculates, for every bit, the expected value of multiplying a Z ~ N(0, 1) by its allocated quantization value
    i.e., E(Z * Q_delta_b(Z))
    """
    if os.path.isfile(ENTROPY_DELTAS_FILE):
        deltas = np.load(ENTROPY_DELTAS_FILE, allow_pickle=True)
    else:
        raise ValueError('precalc_expected_entropy_delta_q_product requires valid deltas')

    if os.path.isfile(ENTROPY_DELTA_CENTROIDS_FILE):
        all_centroids = np.load(ENTROPY_DELTA_CENTROIDS_FILE, allow_pickle=True)
    else:
        raise ValueError('precalc_expected_entropy_delta_q_product requires valid centroids')

    all_eips = {}
    for bits in BITS:
        print(f'Start {bits=}')

        centroids = all_centroids[str(bits)]
        delta = deltas[str(bits)]
        boundaries = [delta * (n - 1 / 2) for n in range(1, len(centroids) + 1)]

        centroids_mass = [norm_c.mass(left, right) for left, right
                          in zip([0, *boundaries], [*boundaries, mp.inf])]

        # multiplying by 2 since we're only going through the positive clusters of the normal distribution
        eip = 2 * sum([m * c ** 2 for c, m in zip([0, *centroids], centroids_mass)])

        all_eips[str(bits)] = eip

    np.savez(ENTROPY_DELTA_EPS_FILE, **all_eips)


def print_entropy_delta_mse():
    all_deltas = {k: v for k, v in np.load(ENTROPY_DELTAS_FILE, allow_pickle=True).items()}
    all_centroids = {k: v for k, v in np.load(ENTROPY_DELTA_CENTROIDS_FILE, allow_pickle=True).items()}

    for bits, delta, centroids in zip(all_deltas.keys(), all_deltas.values(), all_centroids.values()):
        boundaries = [delta * (n - 1 / 2) for n in range(1, len(centroids) + 1)]
        print(f'{bits} bits normal mse = {norm_c.symmetric_full_mse([0, *centroids], boundaries)}')


def print_entropy_deltas_as_literal(deltas):
    all_deltas = {k: v.astype(np.float64) for k, v in deltas.items()}

    print('deltas = {')
    for k, d in all_deltas.items():
        print(f"    {k}: {str(d.item())},")
    print('}')


def print_eps_as_literal(all_eps):
    all_eps = {float(k): v.astype(np.float64) for k, v in all_eps.items()}
    print('eps = {' + ', '.join([f'{k}: {v}' for k, v in all_eps.items()]) + '}')


if __name__ == '__main__':
    """ uncomment below as needed for pre-calculation"""
    # from mpmath import workdps
    # with workdps(DPS):
    #   precalc_expected_entropy_delta_q_product()
    #   precalc_entropy_delta()
    #   precalc_entropy_delta_centroids()

    # print_entropy_delta_mse()
    # print_eps_as_literal(np.load(ENTROPY_DELTA_EPS_FILE, allow_pickle=True))
    print_entropy_deltas_as_literal(np.load(ENTROPY_DELTAS_FILE, allow_pickle=True))
    print('Done')
