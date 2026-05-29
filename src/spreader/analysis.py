"""Aggregation of run ensembles into the observables of the paper."""

import numpy as np


def _select(results, condition):
    """Filter a list of run dicts.

    condition: "all", "percolated", or "outbreak" (final size > 5% of N).
    """
    if condition == "all":
        return results
    if condition == "percolated":
        sel = [r for r in results if r["percolated"]]
    elif condition == "outbreak":
        sel = [r for r in results
               if r["total_infected"] > 0.05 * r["state"].shape[0]]
    else:
        raise ValueError(condition)
    return sel if sel else results          # fall back if nothing matched


def mean_epidemic_curve(results, condition="outbreak"):
    """Mean number of newly infected per timestep (Fig. 8)."""
    sel = _select(results, condition)
    arr = np.array([r["new_counts"] for r in sel], dtype=float)
    return arr.mean(axis=0)


def mean_rf_curve(results, condition="percolated"):
    """Mean front distance r_f / r0 vs time (Fig. 6)."""
    sel = _select(results, condition)
    arr = np.array([r["rf_curve"] for r in sel], dtype=float)
    return arr.mean(axis=0)


def front_velocity(results, condition="percolated", lo=0.1, hi=0.8):
    """Propagation speed (Fig. 7) = slope of the rising part of mean r_f(t).

    Fit a line over the segment where the mean front is between ``lo`` and
    ``hi`` of its plateau value.
    """
    rf = mean_rf_curve(results, condition)
    plateau = rf.max()
    if plateau <= 0:
        return 0.0
    t = np.arange(rf.shape[0], dtype=float)
    mask = (rf >= lo * plateau) & (rf <= hi * plateau)
    if mask.sum() < 2:
        mask = rf < plateau                 # fallback: whole rise
    if mask.sum() < 2:
        return 0.0
    slope = np.polyfit(t[mask], rf[mask], 1)[0]
    return float(slope)


def secondary_distribution(results, max_links=20, include_susceptible=False):
    """Normalised distribution of out-degree over infection-network nodes
    (Figs. 12-13).  Nodes are individuals that were ever infected.
    """
    counts = []
    for r in results:
        sec = r["secondary"]
        if include_susceptible:
            counts.append(sec)
        else:
            counts.append(sec[r["state"] != 0])   # only infected nodes
    counts = np.concatenate(counts)
    edges = np.arange(0, max_links + 2) - 0.5
    hist, _ = np.histogram(counts, bins=edges, density=False)
    hist = hist / hist.sum()
    centres = np.arange(0, max_links + 1)
    return centres, hist


def critical_density(model, lam, X_grid, n_runs, threshold=0.5,
                     density_to_N=None, **batch_kw):
    """Critical reduced density X_c where percolation probability crosses
    ``threshold``, found by scanning ``X_grid`` and linearly interpolating.

    ``density_to_N`` is injected (from models) to avoid a circular import here.
    Returns (X_c, X_grid, prob_grid).
    """
    from .runner import percolation_probability
    if density_to_N is None:
        from .models import density_to_N as density_to_N

    probs = np.array([
        percolation_probability(density_to_N(X), model, lam, n_runs, **batch_kw)
        for X in X_grid
    ])
    X_grid = np.asarray(X_grid, dtype=float)

    # first up-crossing of the threshold
    Xc = np.nan
    for k in range(1, len(X_grid)):
        if probs[k - 1] < threshold <= probs[k]:
            x0, x1 = X_grid[k - 1], X_grid[k]
            p0, p1 = probs[k - 1], probs[k]
            Xc = x0 + (threshold - p0) * (x1 - x0) / (p1 - p0)
            break
    return Xc, X_grid, probs
