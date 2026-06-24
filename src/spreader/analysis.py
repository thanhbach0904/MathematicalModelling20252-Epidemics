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


def run_metrics(results):
    """Per-run scalar metrics for the sensitivity sweeps.

    Returns a dict of arrays (one entry per run): ``peak_time`` (timestep of
    the largest ``new_counts``, 0-indexed), ``peak_magnitude``,
    ``extinction_time`` (= number of sweeps with at least one active
    infective, i.e. ``n_steps``), ``percolated`` and ``total_infected``.
    """
    peak_time = np.zeros(len(results), dtype=float)
    peak_magnitude = np.zeros(len(results), dtype=float)
    extinction_time = np.zeros(len(results), dtype=float)
    percolated = np.zeros(len(results), dtype=bool)
    total_infected = np.zeros(len(results), dtype=float)
    for k, r in enumerate(results):
        n = max(r["n_steps"], 1)
        curve = r["new_counts"][:n]
        peak_time[k] = float(np.argmax(curve))
        peak_magnitude[k] = float(curve.max())
        extinction_time[k] = float(r["n_steps"])
        percolated[k] = r["percolated"]
        total_infected[k] = float(r["total_infected"])
    return {
        "peak_time": peak_time,
        "peak_magnitude": peak_magnitude,
        "extinction_time": extinction_time,
        "percolated": percolated,
        "total_infected": total_infected,
    }


def aggregate_metrics(results):
    """Mean/std of :func:`run_metrics` plus the percolation probability and
    its binomial standard error -- the per-config row consumed by the
    sensitivity-sweep scripts.
    """
    m = run_metrics(results)
    n = len(results)
    p = float(m["percolated"].mean()) if n else float("nan")
    return {
        "peak_time_mean": float(m["peak_time"].mean()),
        "peak_time_std": float(m["peak_time"].std()),
        "peak_magnitude_mean": float(m["peak_magnitude"].mean()),
        "peak_magnitude_std": float(m["peak_magnitude"].std()),
        "extinction_time_mean": float(m["extinction_time"].mean()),
        "extinction_time_std": float(m["extinction_time"].std()),
        "percolation_prob": p,
        "percolation_stderr": float(np.sqrt(p * (1 - p) / n)) if n else float("nan"),
    }


def mse_vs_sars(mean_curve, sars_curve=None, rescale_peak=True):
    """Mean squared error between a mean simulated epidemic curve and the
    SARS Singapore reference curve (default: ``sars_data.SARS_EPIDEMIC_CURVE``),
    padding whichever array is shorter with zeros.

    ``rescale_peak=True`` (default) first rescales the simulated curve so its
    peak matches the SARS curve's peak, exactly as ``scripts/05_sars_comparison.py``
    already does for its plot -- the model population (N~477) is not the
    Singapore population, so raw amplitudes aren't comparable and an
    un-rescaled MSE is dominated by that scale mismatch rather than the
    epidemic *shape*, which is what the paper's comparison is actually about
    (and what's needed for the lambda=0.4 hub-vs-strong MSE ordering in
    CLAUDE.md's validation criterion to hold).
    """
    if sars_curve is None:
        from .sars_data import SARS_EPIDEMIC_CURVE as sars_curve
    a = np.asarray(mean_curve, dtype=float)
    b = np.asarray(sars_curve, dtype=float)
    if rescale_peak and a.max() > 0:
        a = a * (b.max() / a.max())
    n = max(a.shape[0], b.shape[0])
    a = np.pad(a, (0, n - a.shape[0]))
    b = np.pad(b, (0, n - b.shape[0]))
    return float(np.mean((a - b) ** 2))


def sensitivity_index(Q0, Q1, p0, p1):
    """Numerical sensitivity S_hat = ((Q1-Q0)/Q0) / ((p1-p0)/p0)."""
    if Q0 == 0 or p0 == p1:
        return float("nan")
    return ((Q1 - Q0) / Q0) / ((p1 - p0) / p0)


def find_crossing(x_grid, y_grid, target):
    """First x where ``y_grid`` crosses ``target`` (either direction), found
    by linear interpolation between the bracketing grid points. Generalises
    the up-crossing search used by :func:`critical_density`. Returns ``nan``
    if no sign change of ``y - target`` occurs.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    diff = y_grid - target
    for k in range(1, len(x_grid)):
        if diff[k - 1] == 0:
            return float(x_grid[k - 1])
        if diff[k - 1] * diff[k] < 0:
            x0, x1 = x_grid[k - 1], x_grid[k]
            d0, d1 = diff[k - 1], diff[k]
            return float(x0 + (0.0 - d0) * (x1 - x0) / (d1 - d0))
    return float("nan")


def critical_density(model, lam, X_grid, n_runs, threshold=0.5,
                     density_to_N=None, L=None, **batch_kw):
    """Critical reduced density X_c where percolation probability crosses
    ``threshold``, found by scanning ``X_grid`` and linearly interpolating.

    ``density_to_N`` is injected (from models) to avoid a circular import here.
    ``L`` sets the box side (default = the paper's 10 r0); a larger ``L`` keeps
    the reduced density fixed while raising N, which shrinks the finite-size
    shortfall of X_c below the analytic R0=Rc curve.
    Returns (X_c, X_grid, prob_grid).
    """
    from .runner import percolation_probability
    if density_to_N is None:
        from .models import density_to_N as density_to_N
    if L is None:
        from .models import BOX_L as L

    probs = np.array([
        percolation_probability(density_to_N(X, L=L), model, lam, n_runs,
                                L=L, **batch_kw)
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
