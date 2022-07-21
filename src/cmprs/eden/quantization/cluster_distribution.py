from functools import cached_property

from mpmath import mp
from sympy import oo, integrate, simplify, symbols, lambdify
from sympy.stats.crv_types import NormalDistribution


class ClusterDistribution:

    def __init__(self, dist):
        self.dist = dist

    @cached_property
    def mass(self):
        l, r = symbols('l r')
        return lambdify([l, r], simplify(self.dist.cdf(r) - self.dist.cdf(l)), cse=True, modules='mpmath')

    @cached_property
    def centroid(self):
        x, l, r = symbols('x l r')

        return lambdify([l, r], simplify(integrate(x * self.dist.pdf(x), (x, l, r))
                                         / (self.dist.cdf(r) - self.dist.cdf(l))),
                        cse=True,
                        modules='mpmath')

    @cached_property
    def mse(self):
        x, l, r, c = symbols('x l r c')
        mse_exr = lambdify([l, r, c], simplify(integrate((x - c) ** 2 * self.dist.pdf(x), (x, l, r))),
                           cse=True,
                           modules='mpmath')
        # inf is not handled properly when moving to mpmath, so we handle it separately
        mse_exr_inf = lambdify([l, c], simplify(integrate((x - c) ** 2 * self.dist.pdf(x), (x, l, oo))),
                               cse=True,
                               modules='mpmath')

        def _mse_fn(left, right, center):
            if right == mp.inf:
                return mse_exr_inf(left, center)
            else:
                return mse_exr(left, right, center)

        return _mse_fn

    def symmetric_full_mse(self, pos_centers, pos_boundaries):
        # For symmetric distributions, we calculate mse for positive side and multiply by 2
        return 2 * sum([self.mse(left, right, center) for left, right, center in
                        zip([0, *pos_boundaries], [*pos_boundaries, mp.inf], pos_centers)])


normal_cluster_dist = ClusterDistribution(NormalDistribution(0, 1))
