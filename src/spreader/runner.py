"""Batched / parallel execution of Monte-Carlo runs (Tier-1: Numba + joblib)."""

import numpy as np
from joblib import Parallel, delayed

from .models import model_params, W0, GAMMA, BOX_L
from .geometry import sample_positions
from .simulator import run_single, seed_rng


def _make_run(N, model, lam, seed):
    """Build the inputs for one run with reproducible RNG."""
    rng = np.random.default_rng(seed)
    positions = sample_positions(N, BOX_L, rng)
    is_super = (rng.random(N) < lam).astype(np.int8)
    p = model_params(model)
    return positions, is_super, p


def _one_run(seed, N, model, lam, max_steps, perc_threshold, full):
    positions, is_super, p = _make_run(N, model, lam, seed)
    seed_rng(int(seed) & 0x7FFFFFFF)            # seed Numba's RNG per run
    (perc, new_counts, rf_curve, secondary,
     infector, unwrapped, state, n_steps) = run_single(
        positions, is_super,
        p["cutoff_n"], p["exp_n"], p["cutoff_s"], p["exp_s"],
        W0, GAMMA, p["L"], p["n_cells"], p["cell_size"],
        max_steps, perc_threshold)

    total_infected = int(np.count_nonzero(state != 0))
    out = {
        "percolated": bool(perc),
        "total_infected": total_infected,
        "n_steps": int(n_steps),
        "new_counts": new_counts,
        "rf_curve": rf_curve,
        "secondary": secondary,
        "state": state,
        "is_super": is_super,
    }
    if full:
        out["positions"] = positions
        out["unwrapped"] = unwrapped
        out["infector"] = infector
    return out


def run_batch(N, model, lam, n_runs, max_steps=200, perc_threshold=BOX_L,
              n_jobs=-1, base_seed=0, full=False, verbose=False):
    """Run ``n_runs`` independent epidemics; return a list of result dicts."""
    seeds = base_seed + np.arange(n_runs)
    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_one_run)(int(s), N, model, lam, max_steps, perc_threshold, full)
        for s in seeds
    )
    return results


def percolation_probability(N, model, lam, n_runs, **kw):
    """Fraction of runs whose infection front reached the top."""
    res = run_batch(N, model, lam, n_runs, full=False, **kw)
    return float(np.mean([r["percolated"] for r in res]))


def single_full_run(N, model, lam, seed=0, max_steps=200, perc_threshold=BOX_L):
    """One run returning full arrays (positions, infection tree) for plotting."""
    return _one_run(seed, N, model, lam, max_steps, perc_threshold, full=True)
