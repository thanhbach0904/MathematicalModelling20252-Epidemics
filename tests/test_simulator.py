"""Smoke tests + a sanity check of the analytic R0 against simulation.

Run:  python -m pytest tests/ -q     (or: python tests/test_simulator.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spreader.models import (model_params, density_to_N, N_to_density,
                             R0_analytic, critical_density_curve,
                             RC_STRONG, RC_HUB)
from spreader.runner import run_batch, single_full_run


def test_density_population_roundtrip():
    # paper: N = 477 corresponds to rho*pi*r0^2 = 15
    assert density_to_N(15.0) == 477
    assert abs(N_to_density(477) - 15.0) < 0.05


def test_model_params_cell_size_covers_cutoff():
    for model in ("strong", "hub"):
        p = model_params(model)
        assert p["cell_size"] >= max(p["cutoff_n"], p["cutoff_s"]) - 1e-9
        assert p["n_cells"] >= 3


def test_strong_super_is_constant_probability():
    # exponent 0 must give w(r) = w0 across the disk
    p = model_params("strong")
    assert p["exp_s"] == 0.0
    assert p["cutoff_s"] == p["cutoff_n"]


def test_critical_curve_matches_Rc_at_endpoints():
    # X_c(lambda) = 6 Rc / (1 + 5 lambda)
    assert abs(critical_density_curve(1.0, "strong") - RC_STRONG) < 1e-9
    assert abs(critical_density_curve(1.0, "hub") - RC_HUB) < 1e-9


def test_R0_closed_form():
    # R0 = X (1 + 5 lambda) / 6  for both models
    for model in ("strong", "hub"):
        for lam in (0.0, 0.3, 1.0):
            X = 12.0
            expected = X * (1 + 5 * lam) / 6
            assert abs(R0_analytic(lam, model, X=X) - expected) < 1e-9


def test_run_smoke_and_monotonic_outbreak_in_lambda():
    # higher lambda -> larger outbreaks at fixed density (qualitative check)
    N = density_to_N(15.0)
    sizes = []
    for lam in (0.0, 1.0):
        res = run_batch(N, "hub", lam, 40, n_jobs=1, base_seed=1)
        sizes.append(np.mean([r["total_infected"] for r in res]))
    assert sizes[1] > sizes[0]


def test_critical_density_lands_on_R0_eq_Rc_curve():
    # The paper's central percolation claim (Fig. 5): the *measured* critical
    # density coincides with the analytic R0 = Rc curve. With the bottom->top
    # (vertical-spanning) percolation criterion the simulated Xc sits on that
    # curve up to a finite-size (L=10) shortfall, which is largest at lambda=1
    # (critical N ~ 140). Guard the coincidence within that tolerance.
    from spreader.analysis import critical_density
    from spreader.models import critical_density_curve

    X_grid = np.linspace(0.5, 30.0, 25)
    for model in ("strong", "hub"):
        Xc, _, _ = critical_density(model, 1.0, X_grid, 150, n_jobs=-1)
        paper = float(critical_density_curve(1.0, model))
        assert 0.78 * paper <= Xc <= 1.05 * paper, (
            f"{model}: simulated Xc={Xc:.2f} off paper Rc={paper:.2f}")


def test_hub_normalization_holds_for_arbitrary_ratio():
    # closed form: integral_0^c w0*(1-r/c)^2 * 2*pi*r dr = w0*pi*c^2/6 (a=2 case
    # of models._ring_integral). Must equal pi*r0^2 (the strong superspreader's
    # integral) whenever normalize_hub=True, for any hub_ratio -- not just
    # the paper's sqrt(6).
    for ratio in (1.5, 2.0, np.sqrt(6.0), 4.0):
        p = model_params("hub", hub_ratio=ratio, normalize_hub=True)
        integral = p["w0_s"] * np.pi * p["cutoff_s"] ** 2 / 6.0
        assert abs(integral - np.pi) < 1e-9


def test_run_batch_gamma_override_changes_dynamics():
    # lower gamma (slower recovery) should not shrink outbreaks at fixed density
    N = density_to_N(20.0)
    res_fast = run_batch(N, "hub", 0.4, 60, n_jobs=1, base_seed=2, gamma=1.0)
    res_slow = run_batch(N, "hub", 0.4, 60, n_jobs=1, base_seed=2, gamma=0.3)
    size_fast = np.mean([r["total_infected"] for r in res_fast])
    size_slow = np.mean([r["total_infected"] for r in res_slow])
    assert size_slow >= size_fast


def test_single_full_run_returns_tree():
    N = density_to_N(15.0)
    r = single_full_run(N, "hub", 0.4, seed=3)
    assert "infector" in r and "positions" in r
    # every infected non-root node has an infector
    infected = np.where(r["state"] != 0)[0]
    for j in infected:
        if j == 0:
            continue
        assert r["infector"][j] >= 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
