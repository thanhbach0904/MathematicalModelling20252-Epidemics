"""Monte-Carlo SIR dynamics (single run), JIT-compiled with Numba.

Dynamics (one Monte-Carlo step / "sweep"), following Section 2 of the paper:

  * take the snapshot of currently infected individuals;
  * process them in random order. Each infective tries to infect every
    susceptible j within its cutoff with probability w(r_ij). Newly infected
    individuals become infectious only on the *next* sweep ("without new
    infected ones");
  * after acting, the infective recovers with probability gamma.

With gamma = 1 this reduces to a generational (single-shot) SIR process.

Boundary conditions are periodic (a torus). To detect spanning / measure the
propagation front under PBC we carry an *unwrapped* position that accumulates
minimum-image displacements along the infection tree, so the front distance can
legitimately exceed L/sqrt(2).
"""

import numpy as np
from numba import njit

from .geometry import delta
from .cell_list import build_cell_list


@njit(cache=True)
def seed_rng(s):
    """Seed Numba's internal RNG (separate from NumPy's Python-level RNG)."""
    np.random.seed(s)


@njit(cache=True)
def run_single(positions, is_super,
               cutoff_n, exp_n, cutoff_s, exp_s,
               w0, gamma, L, n_cells, cell_size,
               max_steps, perc_threshold, periodic):
    """Run one full epidemic from individual 0.

    ``periodic`` (bool): True = minimum-image torus (phase diagram / critical
    density); False = bounded hard-wall box (no cell wrapping, plain Euclidean
    displacement). On the bounded box ``unwrapped`` coincides with ``positions``
    and ``r_f`` can reach the box scale rather than being capped at L/sqrt(2).

    Returns a tuple:
        percolated   : bool   -- paper's criterion: the infection seeded at the
                                 bottom reaches the top, i.e. the cluster's
                                 unwrapped *vertical* extent reaches
                                 perc_threshold (= L)
        new_counts   : int64[max_steps]   newly infected per sweep (epidemic curve)
        rf_curve     : float64[max_steps] front distance from origin per sweep
        secondary    : int64[N]           out-degree (people each node infected)
        infector     : int64[N]           who infected each node (-1 if none)
        unwrapped    : float64[N, 2]       unwrapped positions
        state        : int8[N]            final state (0=S, 1=I, 2=R)
        n_steps      : int64              number of sweeps actually run
    """
    N = positions.shape[0]
    state = np.zeros(N, dtype=np.int8)             # 0=S, 1=I, 2=R
    unwrapped = positions.copy()
    infector = -np.ones(N, dtype=np.int64)
    secondary = np.zeros(N, dtype=np.int64)
    new_counts = np.zeros(max_steps, dtype=np.int64)
    rf_curve = np.zeros(max_steps, dtype=np.float64)

    head, nxt = build_cell_list(positions, n_cells, cell_size)

    state[0] = 1
    x0 = unwrapped[0, 0]
    y0 = unwrapped[0, 1]
    max_disp = 0.0       # max Euclidean front distance from origin (Fig. 6)
    # unwrapped vertical extent of the infected cluster (bottom->top detection)
    uymin = y0
    uymax = y0

    t = 0
    while t < max_steps:
        infected = np.where(state == 1)[0]
        if infected.shape[0] == 0:
            break
        np.random.shuffle(infected)             # random processing order

        count_new = 0
        for idx in range(infected.shape[0]):
            i = infected[idx]
            if is_super[i] != 0:
                cutoff = cutoff_s
                expo = exp_s
            else:
                cutoff = cutoff_n
                expo = exp_n
            cutoff_sq = cutoff * cutoff
            cx = int(positions[i, 0] / cell_size) % n_cells
            cy = int(positions[i, 1] / cell_size) % n_cells

            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    if periodic:
                        ncx = (cx + ddx) % n_cells
                        ncy = (cy + ddy) % n_cells
                    else:
                        ncx = cx + ddx
                        ncy = cy + ddy
                        if ncx < 0 or ncx >= n_cells or ncy < 0 or ncy >= n_cells:
                            continue
                    j = head[ncx, ncy]
                    while j != -1:
                        if state[j] == 0:
                            dx, dy = delta(positions[i, 0], positions[i, 1],
                                           positions[j, 0], positions[j, 1], L, periodic)
                            d2 = dx * dx + dy * dy
                            if d2 < cutoff_sq:
                                r = np.sqrt(d2)
                                p = w0 * (1.0 - r / cutoff) ** expo
                                if np.random.random() < p:
                                    state[j] = 1
                                    infector[j] = i
                                    secondary[i] += 1
                                    ux = unwrapped[i, 0] + dx
                                    uy = unwrapped[i, 1] + dy
                                    unwrapped[j, 0] = ux
                                    unwrapped[j, 1] = uy
                                    # Front distance r_f (Fig. 6): true geometric
                                    # distance from the seed to this node, NOT the
                                    # path-accumulated unwrapped distance (which
                                    # random-walk drifts over generations and
                                    # inverts the lambda ordering). Use min-image.
                                    sdx, sdy = delta(positions[0, 0], positions[0, 1],
                                                     positions[j, 0], positions[j, 1], L, periodic)
                                    disp = np.sqrt(sdx * sdx + sdy * sdy)
                                    if disp > max_disp:
                                        max_disp = disp
                                    if uy < uymin:
                                        uymin = uy
                                    if uy > uymax:
                                        uymax = uy
                                    count_new += 1
                        j = nxt[j]

            if np.random.random() < gamma:
                state[i] = 2

        new_counts[t] = count_new
        rf_curve[t] = max_disp
        t += 1

    # forward-fill the front plateau so averaging over runs is well-defined
    for k in range(t, max_steps):
        rf_curve[k] = max_disp

    # Paper: a run is percolated when infection started at the bottom reaches the
    # top -- the cluster spans the full height L of the system (vertical extent).
    vertical_span = uymax - uymin
    percolated = vertical_span >= perc_threshold
    return (percolated, new_counts, rf_curve, secondary,
            infector, unwrapped, state, t)
