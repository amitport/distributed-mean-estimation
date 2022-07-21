import torch
from tqdm import tqdm
import cmprs.config as config


def estimate_vNMSE(x, repeats, compression_scheme):
    total_SSE = 0
    for r in range(repeats):
        est_x = compression_scheme.roundtrip(x).rx
        total_SSE += torch.sum((est_x - x) ** 2)
    return total_SSE / repeats / torch.sum(x ** 2)


def estimate_vNMSE_over_dist(n_vecs, repeats_per_vec, sample_vec_fn, compression_scheme):
    total_vNMSE = 0
    for _ in tqdm(range(n_vecs), disable=None):
        x = sample_vec_fn(1)

        total_vNMSE += estimate_vNMSE(x, repeats_per_vec, compression_scheme)
    vNMSE = total_vNMSE / n_vecs
    return vNMSE


def estimate_NMSE(repeats, x_s, compression_scheme):
    n_clients = len(x_s)

    total_x = torch.zeros(x_s.shape[1:], dtype=torch.float64, device=config.device)
    total_sum_squares = 0
    for x in x_s:
        total_x += x
        total_sum_squares += torch.sum(x ** 2)
    avg_x = total_x / n_clients
    avg_sum_squares = total_sum_squares / n_clients

    total_SSE = 0
    for r in range(repeats):
        total_est_x = torch.zeros(x_s.shape[1:], dtype=torch.float64, device=config.device)
        for x in x_s:
            est_x = compression_scheme.roundtrip(x).rx
            total_est_x += est_x

        avg_est_x = total_est_x / n_clients

        total_SSE += torch.sum((avg_x - avg_est_x) ** 2)
    return total_SSE / repeats / avg_sum_squares


def estimate_NMSE_over_dist(n_rounds, n_clients, sample_vec_fn, compression_scheme, repeats_per_round=None):
    # This should equal estimate_vNMSE_over_dist / n_clients when
    # For eden with same quantization intervals
    # See https://arxiv.org/pdf/2108.08842.pdf, Lemma 2.2.

    total_NMSE = 0
    for trial in range(n_rounds):
        if repeats_per_round is not None:
            x_s = sample_vec_fn(n_clients)

            total_NMSE += estimate_NMSE(repeats_per_round, x_s, compression_scheme)
        else:
            # here we assume all clients send the same vector
            # above which generates a vector per client
            vec = sample_vec_fn(1)
            r_vec = torch.zeros_like(vec)
            for client in range(n_clients):
                r_vec += compression_scheme.roundtrip(vec).rx
            r_vec /= n_clients

            total_NMSE += torch.norm(vec - r_vec, 2) ** 2 / torch.norm(vec) ** 2

    NMSE = total_NMSE / n_rounds
    return NMSE
