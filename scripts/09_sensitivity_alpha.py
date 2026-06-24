"""Sensitivity analysis: normal-individual decay exponent alpha (AGENTS.md S6.3).

The paper fixes alpha=2 for normal individuals (superspreader exponents are
unaffected: 0 for strong, 2 for hub). This sweeps alpha in {1, 2, 3} at the
baseline density/gamma and checks whether the hub model keeps a lower
MSE-vs-SARS than the strong model across all three values.

Run:
    python scripts/09_sensitivity_alpha.py --runs 200
"""

import argparse
import csv

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import aggregate_metrics, mean_epidemic_curve, mse_vs_sars
from spreader import visualize as V

MODELS = ["strong", "hub"]
LAMBDAS = [0.2, 0.4]
ALPHAS = [1, 2, 3]
BASELINE_DENSITY = 15.0
BASELINE_GAMMA = 1.0
MSE_LAMBDA = 0.2

RAW_COLS = ["model", "lambda", "alpha", "gamma", "rho_pi_r0sq",
            "peak_time_mean", "peak_time_std",
            "peak_magnitude_mean", "peak_magnitude_std",
            "percolation_prob", "mse_vs_sars"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(BASELINE_DENSITY)
    rows = []
    mse_by = {(m, l): {} for m in MODELS for l in LAMBDAS}

    n_total = len(MODELS) * len(LAMBDAS) * len(ALPHAS)
    k = 0
    for model in MODELS:
        for lam in LAMBDAS:
            for alpha in ALPHAS:
                k += 1
                res = run_batch(N, model, lam, args.runs, gamma=BASELINE_GAMMA,
                                alpha=alpha, n_jobs=args.jobs)
                metrics = aggregate_metrics(res)
                mse = mse_vs_sars(mean_epidemic_curve(res))
                mse_by[(model, lam)][alpha] = mse
                rows.append({
                    "model": model, "lambda": lam, "alpha": alpha,
                    "gamma": BASELINE_GAMMA, "rho_pi_r0sq": BASELINE_DENSITY,
                    "peak_time_mean": metrics["peak_time_mean"],
                    "peak_time_std": metrics["peak_time_std"],
                    "peak_magnitude_mean": metrics["peak_magnitude_mean"],
                    "peak_magnitude_std": metrics["peak_magnitude_std"],
                    "percolation_prob": metrics["percolation_prob"],
                    "mse_vs_sars": mse,
                })
                print(f"[{k}/{n_total}] model={model} lambda={lam} alpha={alpha} "
                      f"-> MSE={mse:.2f} percolation={metrics['percolation_prob']:.2f}")

    raw_path = B.sweep_path("alpha_sweep", "alpha_sweep_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {raw_path}")

    print("\nRobustness table (hub MSE < strong MSE at every alpha?):")
    for lam in LAMBDAS:
        robust = all(mse_by[("hub", lam)][a] < mse_by[("strong", lam)][a] for a in ALPHAS)
        detail = ", ".join(
            f"alpha={a}: hub={mse_by[('hub', lam)][a]:.1f} "
            f"strong={mse_by[('strong', lam)][a]:.1f}" for a in ALPHAS)
        print(f"  lambda={lam}: robust={robust}  ({detail})")

    alpha_arr = np.array(ALPHAS, dtype=float)
    mse_by_model = {m: np.array([mse_by[(m, MSE_LAMBDA)][a] for a in ALPHAS]) for m in MODELS}
    V.plot_mse_curve({m: alpha_arr for m in MODELS}, mse_by_model,
                     xlabel=r"$\alpha$",
                     title=f"MSE vs SARS curve, alpha sweep (lambda={MSE_LAMBDA})",
                     vline=2,
                     savepath=B.report_fig("fig_s5_mse_vs_alpha.png"))
    print("Done. See report/figures/fig_s5_mse_vs_alpha.png")


if __name__ == "__main__":
    main()
