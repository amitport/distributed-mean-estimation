import numpy as np
from mpmath import mp
from tqdm import tqdm

from cmprs.eden.quantization.cluster_distribution import normal_cluster_dist


class LloydMax1DSymmetric:
    def __init__(self, cluster_dist=normal_cluster_dist):
        self.cluster_dist = cluster_dist

    @staticmethod
    def get_boundaries(centers):
        return [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])]

    def get_centers(self, pos_boundaries):
        # We only calculate the positives centers for symmetric distributions
        return [self.cluster_dist.centroid(left, right) for left, right in
                zip([0, *pos_boundaries], [*pos_boundaries, mp.inf])]

    # noinspection PyUnboundLocalVariable
    def run(self, centers, steps, early_stopping, eq_rounds_to_stop):
        """
        :param centers: the current *positive* centers
        """
        prev_centers = centers.astype(np.float64)
        centers = [mp.mpf(_) for _ in centers]
        boundaries = self.get_boundaries(centers)
        first_error = self.cluster_dist.symmetric_full_mse(centers, boundaries)
        errors = [np.float64(first_error)]
        eq_count = 0
        for i in (pbar := tqdm(range(steps))):
            centers = self.get_centers(boundaries)
            curr_err = self.cluster_dist.symmetric_full_mse(centers, boundaries)
            if (i + 1) % 50 == 0:
                errors.append(np.float64(curr_err))
            new_centers = np.asarray(centers, dtype=np.float64)
            if early_stopping and np.all(prev_centers == new_centers):
                eq_count += 1
                if eq_count == eq_rounds_to_stop:
                    print(f'Early stopping: no change in float64 for {eq_count} rounds!')
                    break
            else:
                eq_count = 0
            boundaries = self.get_boundaries(centers)
            prev_centers = new_centers
            pbar.set_description(f'Max-LLoyd (Error = {mp.nstr(curr_err, n=50)}')

        print(f'Stopped after {i + 1} steps')
        print(f'Error diff: {curr_err - first_error}')
        print(f'Max 50-steps Float64 error decrease: {max([a - b for a, b in zip(errors[:-1], errors[1:])])}')
        return centers, errors
