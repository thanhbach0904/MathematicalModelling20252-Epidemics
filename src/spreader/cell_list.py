"""Linked-cell list for O(N) neighbour lookup on a periodic box.

The box is divided into ``n_cells`` x ``n_cells`` square cells of side
``cell_size``. ``cell_size`` must be >= the largest interaction cutoff so that
every neighbour within the cutoff lives in one of the 3x3 cells around the
source cell.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def build_cell_list(positions, n_cells, cell_size):
    """Return (head, nxt) linked-list arrays.

    ``head[cx, cy]`` is the index of the first individual in cell (cx, cy) or
    -1. ``nxt[i]`` is the next individual in the same cell as ``i`` or -1.
    """
    N = positions.shape[0]
    head = -np.ones((n_cells, n_cells), dtype=np.int64)
    nxt = -np.ones(N, dtype=np.int64)
    for i in range(N):
        cx = int(positions[i, 0] / cell_size) % n_cells
        cy = int(positions[i, 1] / cell_size) % n_cells
        nxt[i] = head[cx, cy]
        head[cx, cy] = i
    return head, nxt
