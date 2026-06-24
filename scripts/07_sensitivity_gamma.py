"""Sensitivity analysis: recovery probability gamma (AGENTS.md S6.1).

The paper fixes gamma = 1 (a generational process). Realistic SARS values
(1 timestep = 6 days, infectious period ~12 days) give gamma ~ 0.5. This
sweeps model x lambda x gamma x density, recording the same per-config
metrics as the other sweeps, and checks whether the hub model's MSE-vs-SARS
advantage over the strong model survives as gamma drops away from 1.

Run:
    python scripts/07_sensitivity_gamma.py --runs 200
"""

import argparse
import csv

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import aggregate_metrics, mean_epidemic_curve, mse_vs_sars, \
    sensitivity_index, find_crossing
from spreader import visualize as V

MODELS = ["strong", "hub"]
LAMBDAS = [0.0, 0.2, 0.4]
GAMMAS = [0.3, 0.5, 0.7, 1.0]
DENSITIES = [10, 15, 20]
MSE_LAMBDA, MSE_DENSITY = 0.4, 15

RAW_COLS = ["model", "lambda", "gamma", "rho_pi_r0sq", "N",
            "peak_time_mean", "peak_time_std",
            "peak_magnitude_mean", "peak_magnitude_std",
            "extinction_time_mean", "extinction_time_std",
            "percolation_prob", "percolation_stderr", "mse_vs_sars"]

SENS_COLS = ["model", "lambda", "rho_pi_r0sq", "output_metric",
             "baseline_gamma", "baseline_value",
             "perturbed_gamma", "perturbed_value", "S_hat"]

OUTPUT_METRICS = ["peak_time_mean", "peak_magnitude_mean", "extinction_time_mean",
                  "percolation_prob"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    rows = []
    curves_for_fig = {}          # (model, gamma) -> mean curve, at MSE_LAMBDA/MSE_DENSITY
    mse_for_fig = {m: {g: None for g in GAMMAS} for m in MODELS}

    n_total = len(MODELS) * len(LAMBDAS) * len(GAMMAS) * len(DENSITIES)
    k = 0
    for model in MODELS:
        for lam in LAMBDAS:
            for gamma in GAMMAS:
                for density in DENSITIES:
                    k += 1
                    N = density_to_N(density)
                    res = run_batch(N, model, lam, args.runs, gamma=gamma,
                                    n_jobs=args.jobs)
                    metrics = aggregate_metrics(res)
                    mse = float("nan")
                    if lam == MSE_LAMBDA and density == MSE_DENSITY:
                        curve = mean_epidemic_curve(res)
                        mse = mse_vs_sars(curve)
                        curves_for_fig[(model, gamma)] = curve
                        mse_for_fig[model][gamma] = mse
                    rows.append({
                        "model": model, "lambda": lam, "gamma": gamma,
                        "rho_pi_r0sq": density, "N": N,
                        "mse_vs_sars": mse, **metrics,
                    })
                    print(f"[{k}/{n_total}] model={model} lambda={lam} gamma={gamma} "
                          f"rho={density} N={N} -> percolation={metrics['percolation_prob']:.2f} "
                          f"peak_mag={metrics['peak_magnitude_mean']:.1f}"
                          + (f" MSE={mse:.1f}" if mse == mse else ""))

    raw_path = B.sweep_path("gamma_sweep", "gamma_sweep_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {raw_path}")

    # --- sensitivity table: baseline gamma=1.0 vs nearest available grid
    # point below it (0.7). The grid has no literal +-10% points around 1.0,
    # so this departs from the literal AGENTS.md wording -- see plan notes.
    baseline_gamma, perturbed_gamma = 1.0, 0.7
    sens_rows = []
    by_key = {(r["model"], r["lambda"], r["rho_pi_r0sq"], r["gamma"]): r for r in rows}
    for model in MODELS:
        for lam in LAMBDAS:
            for density in DENSITIES:
                r0 = by_key[(model, lam, density, baseline_gamma)]
                r1 = by_key[(model, lam, density, perturbed_gamma)]
                for metric in OUTPUT_METRICS:
                    s_hat = sensitivity_index(r0[metric], r1[metric],
                                              baseline_gamma, perturbed_gamma)
                    sens_rows.append({
                        "model": model, "lambda": lam, "rho_pi_r0sq": density,
                        "output_metric": metric,
                        "baseline_gamma": baseline_gamma, "baseline_value": r0[metric],
                        "perturbed_gamma": perturbed_gamma, "perturbed_value": r1[metric],
                        "S_hat": s_hat,
                    })
    sens_path = B.sweep_path("gamma_sweep", "gamma_sensitivity.csv")
    with open(sens_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SENS_COLS)
        w.writeheader()
        w.writerows(sens_rows)
    print(f"Wrote {len(sens_rows)} rows -> {sens_path}")

    # --- figures ---
    V.plot_gamma_epidemic_curves(curves_for_fig, GAMMAS,
                                 savepath=B.report_fig("fig_s1_gamma_sweep.png"))
    gamma_arr = np.array(GAMMAS, dtype=float)
    mse_by_model = {m: np.array([mse_for_fig[m][g] for g in GAMMAS]) for m in MODELS}
    strong_baseline_mse = mse_by_model["strong"][GAMMAS.index(1.0)]
    crossing = find_crossing(gamma_arr, mse_by_model["hub"], strong_baseline_mse)
    V.plot_mse_curve({m: gamma_arr for m in MODELS}, mse_by_model,
                     xlabel=r"$\gamma$",
                     title=f"MSE vs SARS curve, gamma sweep (lambda={MSE_LAMBDA}, "
                           f"rho={MSE_DENSITY})",
                     baseline=strong_baseline_mse,
                     baseline_label="strong @ gamma=1.0",
                     crossing=crossing,
                     savepath=B.report_fig("fig_s2_mse_vs_gamma.png"))

    if np.isfinite(crossing):
        print(f"\nRobustness boundary: hub MSE crosses strong baseline MSE "
              f"at gamma={crossing:.3f}")
    else:
        print("\nNo gamma in [0.3, 1.0] where hub MSE crosses the strong "
              "baseline MSE -- hub stays better (or worse) across the whole grid.")
    print("Done. See report/figures/fig_s1_gamma_sweep.png and fig_s2_mse_vs_gamma.png")


if __name__ == "__main__":
    main()
