"""Morozov's discrepancy principle: read the prox weight off the noise level"""
import csv
import math
import os

import torch


def resid2(fm, x, y):
    # fm.A, not fm(x): the operator's __call__ adds a noise
    return float(((fm.A(x) - y) ** 2).sum())


def n_observed(fm, x_like):
    """m is the number of measurements that is observed"""
    m = getattr(fm, "_n_observed", None)
    if m is None:
        with torch.no_grad():
            m = int(fm.A(torch.ones_like(x_like)).count_nonzero())
        fm._n_observed = m
    return m


def apply_prox(fm, u, y, gamma):
    # gamma == 0 means the residual is already at or below the floor
    return u if float(gamma) <= 0 else fm.prox_l2(u, y=y, gamma=gamma)


def morozov_gamma(fm, u, y, target, iters=30, gmin=1e-6, gmax=1e8, tol=2e-3):
    """Solve R(gamma) = target for gamma. Returns (gamma, n_evals).
    R(gamma) = sum_j r_j^2 / (1 + gamma s_j^2)^2"""

    r0 = resid2(fm, u, y)
    if r0 <= target:  # already below the floor; any prox-only overshoots
        return 0.0, 0

    log_target = math.log(target)
    n_ev = 0

    def f(log_gamma):
        nonlocal n_ev
        n_ev += 1
        r = resid2(fm, apply_prox(fm, u, y, math.exp(log_gamma)), y)
        return math.log(max(r, 1e-30)) - log_target

    # r0 > target above, so fa > 0: gmin always brackets from the high side
    a, fa = math.log(gmin), math.log(r0) - log_target
    b, fb = math.log(gmax), f(math.log(gmax))
    if fb >= 0:  # target unreachable even at gmax: the operator's null space
        return gmax, n_ev

    for _ in range(iters):
        c = min(max((a * fb - b * fa) / (fb - fa), a + 1e-9), b - 1e-9)
        fc = f(c)
        if abs(fc) < tol:
            return math.exp(c), n_ev
        if fc > 0:
            a, fa = c, fc
            fb *= 0.5  # Illinois: halve the stale side so neither end can stall
        else:
            b, fb = c, fc
            fa *= 0.5
    return math.exp(0.5 * (a + b)), n_ev


def log_step(logdir, row):
    """One row per solver step: rho = ||Ax_k-y||^2 / (m sigma^2), the residual in noise units."""
    path = os.path.join(logdir, "rho.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["k", "t", "gamma", "r_pre", "r_post", "target", "rho", "n_ev"])
        w.writerow(row)
