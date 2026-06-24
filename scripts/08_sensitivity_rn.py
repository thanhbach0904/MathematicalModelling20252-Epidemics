"""Sensitivity analysis: hub range ratio r_n/r0 (AGENTS.md S6.2).

Tests whether there is a critical r_n/r0 below which the hub model loses its
fit advantage over the strong-infectiousness model. For each ratio, runs
both the paper's normalized hub (w0_s rescaled so the infection integral
matches the strong superspreader's) and an intentionally un-normalized
version (w0_s = w0 fixed, so a larger radius means strictly more infection
capacity).

Run:
    python scripts/08_sensitivity_rn.py --runs 200
"""

import argparse
import csv

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import aggregate_metrics, mean_epidemic_curve, mse_vs_sars, \
    find_crossing
from spreader import visualize as V

LAMBDAS = [0.2, 0.4]
RATIOS = [1.1, 1.5, 2.0, np.sqrt(6.0), 3.0, 4.0, 5.0]
BASELINE_DENSITY = 15.0
BASELINE_GAMMA = 1.0

RAW_COLS = ["model", "lambda", "rn_r0_ratio", "normalized", "gamma", "rho_pi_r0sq",
            "peak_time_mean", "peak_time_std",
            "peak_magnitude_mean", "peak_magnitude_std",
            "extinction_time_mean", "extinction_time_std",
            "percolation_prob", "mse_vs_sars"]

CRIT_COLS = ["lambda", "normalized", "critical_rn_r0_ratio"]


def _row(model, lam, rn_ratio, normalized, N, res):
    metrics = aggregate_metrics(res)
    mse = mse_vs_sars(mean_epidemic_curve(res))
    return {
        "model": model, "lambda": lam, "rn_r0_ratio": rn_ratio,
        "normalized": normalized, "gamma": BASELINE_GAMMA,
        "rho_pi_r0sq": BASELINE_DENSITY,
        "peak_time_mean": metrics["peak_time_mean"],
        "peak_time_std": metrics["peak_time_std"],
        "peak_magnitude_mean": metrics["peak_magnitude_mean"],
        "peak_magnitude_std": metrics["peak_magnitude_std"],
        "extinction_time_mean": metrics["extinction_time_mean"],
        "extinction_time_std": metrics["extinction_time_std"],
        "percolation_prob": metrics["percolation_prob"],
        "mse_vs_sars": mse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(BASELINE_DENSITY)
    rows = []

    # strong-model reference (rn_r0_ratio irrelevant): one row per lambda
    strong_mse_by_lambda = {}
    for lam in LAMBDAS:
        res = run_batch(N, "strong", lam, args.runs, gamma=BASELINE_GAMMA, n_jobs=args.jobs)
        row = _row("strong", lam, float("nan"), None, N, res)
        rows.append(row)
        strong_mse_by_lambda[lam] = row["mse_vs_sars"]
        print(f"strong reference: lambda={lam} -> MSE={row['mse_vs_sars']:.2f}")

    mse_grid = {lam: {True: [], False: []} for lam in LAMBDAS}
    n_total = len(LAMBDAS) * len(RATIOS) * 2
    k = 0
    for lam in LAMBDAS:
        for ratio in RATIOS:
            for normalized in (True, False):
                k += 1
                res = run_batch(N, "hub", lam, args.runs, gamma=BASELINE_GAMMA,
                                hub_ratio=ratio, normalize_hub=normalized,
                                n_jobs=args.jobs)
                row = _row("hub", lam, ratio, normalized, N, res)
                rows.append(row)
                mse_grid[lam][normalized].append(row["mse_vs_sars"])
                print(f"[{k}/{n_total}] hub lambda={lam} rn_ratio={ratio:.3f} "
                      f"normalized={normalized} -> MSE={row['mse_vs_sars']:.2f}")

    raw_path = B.sweep_path("rn_sweep", "rn_sweep_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {raw_path}")

    ratio_arr = np.array(RATIOS, dtype=float)
    crit_rows = []
    crossings = {}
    for lam in LAMBDAS:
        for normalized in (True, False):
            mse_arr = np.array(mse_grid[lam][normalized])
            crossing = find_crossing(ratio_arr, mse_arr, strong_mse_by_lambda[lam])
            crossings[(lam, normalized)] = crossing
            crit_rows.append({"lambda": lam, "normalized": normalized,
                              "critical_rn_r0_ratio": crossing})
    crit_path = B.sweep_path("rn_sweep", "critical_ratio.csv")
    with open(crit_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CRIT_COLS)
        w.writeheader()
        w.writerows(crit_rows)
    print(f"Wrote {len(crit_rows)} rows -> {crit_path}")

    # figure for the paper's headline comparison lambda
    fig_lam = 0.4 if 0.4 in LAMBDAS else LAMBDAS[-1]
    V.plot_rn_sweep(ratio_arr, mse_grid[fig_lam][True], mse_grid[fig_lam][False],
                    strong_mse_by_lambda[fig_lam], fig_lam,
                    crossing_normalized=crossings[(fig_lam, True)],
                    crossing_unnormalized=crossings[(fig_lam, False)],
                    savepath=B.report_fig("fig_s3_mse_vs_rn.png"))

    for (lam, normalized), c in crossings.items():
        tag = "normalized" if normalized else "unnormalized"
        if np.isfinite(c):
            print(f"lambda={lam} ({tag}): hub MSE crosses strong MSE at rn/r0={c:.3f}")
        else:
            print(f"lambda={lam} ({tag}): no crossing in [{RATIOS[0]}, {RATIOS[-1]}]")
    print("Done. See report/figures/fig_s3_mse_vs_rn.png")


if __name__ == "__main__":
    main()
