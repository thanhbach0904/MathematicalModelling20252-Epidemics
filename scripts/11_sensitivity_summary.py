"""Roll the four sweeps (07-10) into the summary tables of AGENTS.md S9.

Run after scripts 07-10 have produced their CSVs:
    python scripts/11_sensitivity_summary.py
"""

import csv
import os

import _bootstrap as B
import numpy as np

from spreader.analysis import sensitivity_index, find_crossing

SUMMARY_COLS = ["experiment", "parameter", "baseline_value", "output_metric",
                "S_hat_peak_time", "S_hat_peak_magnitude", "S_hat_extinction_time",
                "conclusion_robust"]
BOUNDARY_COLS = ["parameter", "boundary_value", "boundary_type", "interpretation"]


def _load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("", None):
                r[k] = float("nan")
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass   # keep strings (model name, "True"/"False"/"nan" stays str only if not castable)
    return rows


def _avg_abs_shat(rows, metric_col, baseline, perturbed, group_keys):
    """Average |S_hat| for ``metric_col`` over every group in ``group_keys``
    (e.g. every (model, lambda, rho) combo), comparing the rows at parameter
    value ``baseline`` vs ``perturbed`` within the same group.
    """
    by_key = {}
    for r in rows:
        by_key.setdefault(tuple(r[k] for k in group_keys), {})[r["__param__"]] = r
    vals = []
    for grp in by_key.values():
        if baseline in grp and perturbed in grp:
            s = sensitivity_index(grp[baseline][metric_col], grp[perturbed][metric_col],
                                  baseline, perturbed)
            if s == s:   # not NaN
                vals.append(abs(s))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    summary_rows = []
    boundary_rows = []

    # --- gamma ---
    gamma_raw = _load(B.sweep_path("gamma_sweep", "gamma_sweep_raw.csv"))
    for r in gamma_raw:
        r["__param__"] = r["gamma"]
    baseline, perturbed = 1.0, 0.7
    group_keys = ["model", "lambda", "rho_pi_r0sq"]
    s_pt = _avg_abs_shat(gamma_raw, "peak_time_mean", baseline, perturbed, group_keys)
    s_pm = _avg_abs_shat(gamma_raw, "peak_magnitude_mean", baseline, perturbed, group_keys)
    s_et = _avg_abs_shat(gamma_raw, "extinction_time_mean", baseline, perturbed, group_keys)
    mse_rows = [r for r in gamma_raw if r["lambda"] == 0.4 and r["rho_pi_r0sq"] == 15
               and r["mse_vs_sars"] == r["mse_vs_sars"]]
    hub_mse = {r["gamma"]: r["mse_vs_sars"] for r in mse_rows if r["model"] == "hub"}
    strong_mse = {r["gamma"]: r["mse_vs_sars"] for r in mse_rows if r["model"] == "strong"}
    robust = hub_mse.get(perturbed, float("nan")) < strong_mse.get(perturbed, float("nan"))
    summary_rows.append({"experiment": "gamma_sweep", "parameter": "gamma",
                         "baseline_value": baseline, "output_metric": "all",
                         "S_hat_peak_time": s_pt, "S_hat_peak_magnitude": s_pm,
                         "S_hat_extinction_time": s_et, "conclusion_robust": robust})

    gammas_sorted = sorted(strong_mse)
    strong_arr = np.array([strong_mse[g] for g in gammas_sorted])
    hub_arr = np.array([hub_mse[g] for g in gammas_sorted])
    # crossing of hub MSE against the strong baseline (gamma=1.0) MSE level
    crossing = find_crossing(np.array(gammas_sorted), hub_arr, strong_mse.get(1.0, float("nan")))
    boundary_rows.append({
        "parameter": "gamma", "boundary_value": crossing,
        "boundary_type": "hub MSE crosses strong MSE (lambda=0.4, rho=15)",
        "interpretation": (f"hub stops being the better SARS fit below gamma={crossing:.3f}"
                           if crossing == crossing else
                           "no crossing in the swept gamma range [0.3, 1.0]; "
                           "the hub-vs-strong ranking holds throughout"),
    })

    # --- alpha ---
    alpha_raw = _load(B.sweep_path("alpha_sweep", "alpha_sweep_raw.csv"))
    for r in alpha_raw:
        r["__param__"] = r["alpha"]
    group_keys = ["model", "lambda"]
    for perturbed in (1.0, 3.0):
        s_pt = _avg_abs_shat(alpha_raw, "peak_time_mean", 2.0, perturbed, group_keys)
        s_pm = _avg_abs_shat(alpha_raw, "peak_magnitude_mean", 2.0, perturbed, group_keys)
        rows_at_p = [r for r in alpha_raw if r["alpha"] == perturbed]
        hub_p = {r["lambda"]: r["mse_vs_sars"] for r in rows_at_p if r["model"] == "hub"}
        strong_p = {r["lambda"]: r["mse_vs_sars"] for r in rows_at_p if r["model"] == "strong"}
        robust = all(hub_p[l] < strong_p[l] for l in hub_p)
        summary_rows.append({"experiment": "alpha_sweep", "parameter": "alpha",
                             "baseline_value": 2.0, "output_metric": "all",
                             "S_hat_peak_time": s_pt, "S_hat_peak_magnitude": s_pm,
                             "S_hat_extinction_time": float("nan"),
                             "conclusion_robust": robust})

    # --- rn ratio ---
    rn_raw = _load(B.sweep_path("rn_sweep", "rn_sweep_raw.csv"))
    crit_rows = _load(B.sweep_path("rn_sweep", "critical_ratio.csv"))
    for r in rn_raw:
        r["__param__"] = r["rn_r0_ratio"]
    sqrt6 = np.sqrt(6.0)

    def _is_norm(r, want):
        v = r["normalized"]
        return isinstance(v, str) and (v == "True") == want

    for normalized in (True, False):
        hub_rows = [r for r in rn_raw if r["model"] == "hub" and _is_norm(r, normalized)]
        group_keys = ["lambda"]
        nearest = min((r["rn_r0_ratio"] for r in hub_rows if r["rn_r0_ratio"] != sqrt6),
                      key=lambda x: abs(x - sqrt6), default=float("nan"))
        s_pm = _avg_abs_shat(hub_rows, "peak_magnitude_mean", sqrt6, nearest, group_keys) \
            if nearest == nearest else float("nan")
        tag = "normalized" if normalized else "unnormalized"
        match = [c for c in crit_rows if c["lambda"] == 0.4 and _is_norm(c, normalized)]
        crossing = match[0]["critical_rn_r0_ratio"] if match else float("nan")
        summary_rows.append({"experiment": "rn_sweep", "parameter": f"rn_r0_ratio ({tag})",
                             "baseline_value": sqrt6, "output_metric": "peak_magnitude",
                             "S_hat_peak_time": float("nan"), "S_hat_peak_magnitude": s_pm,
                             "S_hat_extinction_time": float("nan"),
                             "conclusion_robust": not (crossing == crossing)})
        boundary_rows.append({
            "parameter": f"rn_r0_ratio_{tag}", "boundary_value": crossing,
            "boundary_type": "hub MSE crosses strong MSE (lambda=0.4)",
            "interpretation": (f"hub advantage disappears below rn/r0={crossing:.3f} ({tag})"
                               if crossing == crossing else
                               f"no crossing found for rn/r0 in [1.1, 5.0] ({tag}); "
                               "hub stays better (or worse) across the whole range"),
        })

    # --- lambda ---
    lam_raw = _load(B.sweep_path("lambda_sweep", "small_lambda_sweep.csv"))
    mse_cmp = _load(B.sweep_path("lambda_sweep", "lambda_mse_comparison.csv"))
    for r in lam_raw:
        r["__param__"] = r["lambda"]
    lambdas_sorted = sorted({r["lambda"] for r in lam_raw})
    i = lambdas_sorted.index(0.025) if 0.025 in lambdas_sorted else 2
    lo, hi = lambdas_sorted[max(i - 1, 0)], lambdas_sorted[min(i + 1, len(lambdas_sorted) - 1)]
    group_keys = ["model"]
    s_pt = _avg_abs_shat(lam_raw, "peak_time_mean", lo, hi, group_keys)
    s_pm = _avg_abs_shat(lam_raw, "peak_magnitude_mean", lo, hi, group_keys)
    min_lambda = float("nan")
    for lam in lambdas_sorted:
        hub_rank = next(c["rank"] for c in mse_cmp if c["lambda"] == lam and c["model"] == "hub")
        if hub_rank == 1.0:
            min_lambda = lam
            break
    summary_rows.append({"experiment": "lambda_sweep", "parameter": "lambda",
                         "baseline_value": 0.025, "output_metric": "all",
                         "S_hat_peak_time": s_pt, "S_hat_peak_magnitude": s_pm,
                         "S_hat_extinction_time": float("nan"),
                         "conclusion_robust": min_lambda == min_lambda and min_lambda <= 0.025})
    boundary_rows.append({
        "parameter": "lambda", "boundary_value": min_lambda,
        "boundary_type": "minimum lambda for hub advantage",
        "interpretation": (f"hub fits SARS better than strong for lambda >= {min_lambda} "
                           "(smallest swept value where this held)"
                           if min_lambda == min_lambda else
                           "hub never fits SARS better than strong in [0.01, 0.20]"),
    })

    summary_path = os.path.join(B.RESULTS, "sensitivity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows -> {summary_path}")

    boundary_path = os.path.join(B.RESULTS, "robustness_boundaries.csv")
    with open(boundary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOUNDARY_COLS)
        w.writeheader()
        w.writerows(boundary_rows)
    print(f"Wrote {len(boundary_rows)} rows -> {boundary_path}")

    print("\n--- Robustness boundaries ---")
    for r in boundary_rows:
        print(f"  {r['parameter']}: {r['interpretation']}")


if __name__ == "__main__":
    main()
