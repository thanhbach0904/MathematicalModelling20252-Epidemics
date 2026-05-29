"""Geometry helpers: position sampling and minimum-image displacement (PBC)."""

import numpy as np
from numba import njit


def sample_positions(N, L, rng, init_at_bottom=True):
    """Place ``N`` individuals uniformly on the L x L box.

    Index 0 is the initial-infected individual; per the paper it sits on the
    bottom edge (we use the bottom-centre point).
    """
    pos = rng.random((N, 2)) * L
    if init_at_bottom:
        pos[0, 0] = 0.5 * L
        pos[0, 1] = 0.0
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
