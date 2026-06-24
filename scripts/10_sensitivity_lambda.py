"""Sensitivity analysis: small superspreader fraction lambda (AGENTS.md S6.4).

The paper only explores lambda >= 0.2, but real SARS Singapore had ~3.2%
documented superspreaders. This sweeps lambda down to 0.01 at the baseline
density/gamma and finds the minimum lambda at which the hub model fits the
SARS curve better than the strong model.

Run:
    python scripts/10_sensitivity_lambda.py --runs 200
"""

import argparse
import csv

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import aggregate_metrics, mean_epidemic_curve, mse_vs_sars, \
    sensitivity_index
from spreader import visualize as V

MODELS = ["strong", "hub"]
LAMBDAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
BASELINE_DENSITY = 15.0
BASELINE_GAMMA = 1.0
REALISTIC_LAMBDA = 0.03

RAW_COLS = ["model", "lambda", "gamma", "rho_pi_r0sq", "N",
            "peak_time_mean", "peak_time_std",
            "peak_magnitude_mean", "peak_magnitude_std",
            "percolation_prob", "mse_vs_sars"]

MSE_COLS = ["model", "lambda", "MSE", "MSE_stderr", "rank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(BASELINE_DENSITY)
    rows = []
    mse_by_model = {m: [] for m in MODELS}
    mse_runs_by_model = {m: {} for m in MODELS}   # for a bootstrap stderr

    n_total = len(MODELS) * len(LAMBDAS)
    k = 0
    for model in MODELS:
        for lam in LAMBDAS:
            k += 1
            res = run_batch(N, model, lam, args.runs, gamma=BASELINE_GAMMA,
                            n_jobs=args.jobs)
            metrics = aggregate_metrics(res)
            curve = mean_epidemic_curve(res)
            mse = mse_vs_sars(curve)
            mse_by_model[model].append(mse)

            # bootstrap stderr of the MSE over the run ensemble
            curves = np.array([r["new_counts"] for r in res], dtype=float)
            boot = []
            rng = np.random.default_rng(0)
            for _ in range(200):
                sample = curves[rng.integers(0, curves.shape[0], curves.shape[0])]
                boot.append(mse_vs_sars(sample.mean(axis=0)))
            mse_runs_by_model[(model, lam)] = float(np.std(boot))

            rows.append({
                "model": model, "lambda": lam, "gamma": BASELINE_GAMMA,
                "rho_pi_r0sq": BASELINE_DENSITY, "N": N,
                "peak_time_mean": metrics["peak_time_mean"],
                "peak_time_std": metrics["peak_time_std"],
                "peak_magnitude_mean": metrics["peak_magnitude_mean"],
                "peak_magnitude_std": metrics["peak_magnitude_std"],
                "percolation_prob": metrics["percolation_prob"],
                "mse_vs_sars": mse,
            })
            print(f"[{k}/{n_total}] model={model} lambda={lam} -> MSE={mse:.2f}")

    raw_path = B.sweep_path("lambda_sweep", "small_lambda_sweep.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {raw_path}")

    mse_rows = []
    for lam in LAMBDAS:
        mses = {m: mse_by_model[m][LAMBDAS.index(lam)] for m in MODELS}
        ranked = sorted(MODELS, key=lambda m: mses[m])
        rank = {m: i + 1 for i, m in enumerate(ranked)}
        for m in MODELS:
            mse_rows.append({"model": m, "lambda": lam, "MSE": mses[m],
                             "MSE_stderr": mse_runs_by_model[(m, lam)],
                             "rank": rank[m]})
    mse_path = B.sweep_path("lambda_sweep", "lambda_mse_comparison.csv")
    with open(mse_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MSE_COLS)
        w.writeheader()
        w.writerows(mse_rows)
    print(f"Wrote {len(mse_rows)} rows -> {mse_path}")

    # minimum lambda at which hub beats strong
    min_lambda = float("nan")
    for lam in LAMBDAS:
        i = LAMBDAS.index(lam)
        if mse_by_model["hub"][i] < mse_by_model["strong"][i]:
            min_lambda = lam
            break
    print(f"\nMinimum lambda threshold (hub MSE < strong MSE): {min_lambda}")

    # S_hat(MSE, lambda) for hub near the realistic SARS estimate (0.03),
    # using the adjacent grid points (0.02 -> 0.05) since 0.03 isn't itself
    # flanked by +-10% grid points.
    if REALISTIC_LAMBDA in LAMBDAS:
        i = LAMBDAS.index(REALISTIC_LAMBDA)
        lo, hi = LAMBDAS[max(i - 1, 0)], LAMBDAS[min(i + 1, len(LAMBDAS) - 1)]
        for model in MODELS:
            mse_lo = mse_by_model[model][LAMBDAS.index(lo)]
            mse_hi = mse_by_model[model][LAMBDAS.index(hi)]
            s_hat = sensitivity_index(mse_lo, mse_hi, lo, hi)
            print(f"  S_hat(MSE,lambda) [{model}] between lambda={lo} and {hi}: {s_hat:.3f}")

    lam_arr = np.array(LAMBDAS, dtype=float)
    V.plot_mse_curve({m: lam_arr for m in MODELS},
                     {m: np.array(mse_by_model[m]) for m in MODELS},
                     xlabel=r"$\lambda$",
                     title="MSE vs SARS curve, small lambda sweep",
                     vline=REALISTIC_LAMBDA,
                     savepath=B.report_fig("fig_s4_mse_vs_lambda.png"))
    print("Done. See report/figures/fig_s4_mse_vs_lambda.png")


if __name__ == "__main__":
    main()
