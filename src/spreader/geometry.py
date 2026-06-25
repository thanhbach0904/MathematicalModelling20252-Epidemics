"""Geometry helpers: position sampling and minimum-image displacement (PBC)."""

import numpy as np
from numba import njit


def sample_positions(N, L, rng, init_mode="bottom-random", init_at_bottom=None):
    """Place ``N`` individuals uniformly on the L x L box.

    Each call resamples the whole population for one independent Monte Carlo
    run. Index 0 is the initial-infected individual. By default it is placed on
    the bottom edge with a random x-coordinate, matching the paper's "bottom of
    the system" setup without fixing the same patient location in every run.

    ``init_mode``:
        ``"bottom-random"``: patient zero is random along the bottom edge.
        ``"bottom-center"``: legacy behaviour, fixed at the bottom centre.
        ``"uniform"``: patient zero is sampled uniformly in the full box.
    """
    if init_at_bottom is not None:
        init_mode = "bottom-random" if init_at_bottom else "uniform"

    pos = rng.random((N, 2)) * L
    if init_mode == "bottom-random":
        pos[0, 1] = 0.0
    elif init_mode == "bottom-center":
        pos[0, 0] = 0.5 * L
        pos[0, 1] = 0.0
    elif init_mode == "uniform":
        pass
    else:
        raise ValueError(f"unknown init_mode {init_mode!r}")
    return np.ascontiguousarray(pos, dtype=np.float64)


@njit(cache=True)
def min_image_delta(xi, yi, xj, yj, L):
    """Vector j-i under the minimum-image convention (periodic boundaries)."""
    dx = xj - xi
    dy = yj - yi
    if dx > 0.5 * L:
        dx -= L
    elif dx < -0.5 * L:
        dx += L
    if dy > 0.5 * L:
        dy -= L
    elif dy < -0.5 * L:
        dy += L
    return dx, dy
