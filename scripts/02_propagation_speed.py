"""Figs. 6-7: front distance r_f(t) and propagation velocity vs lambda.

Uses rho*pi*r0^2 = 20 (as in the paper, Fig. 7).

Run:
    python scripts/02_propagation_speed.py --runs 300
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import mean_rf_curve, front_velocity
from spreader import visualize as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--density", type=float, default=20.0,
                    help="rho*pi*r0^2")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(args.density)
    print(f"rho*pi*r0^2 = {args.density} -> N = {N}")

    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Fig. 6: r_f(t) for the strong model
    rf_by_lambda = {}
    vel_by_model = {"strong": [], "hub": []}
    for model in ("strong", "hub"):
        print(f"\n=== {model} model ===")
        for lam in lam_grid:
            res = run_batch(N, model, lam, args.runs, n_jobs=args.jobs)
            rf = mean_rf_curve(res, condition="percolated")
            v = front_velocity(res, condition="percolated")
            vel_by_model[model].append(v)
            if model == "strong":
                rf_by_lambda[lam] = rf
            print(f"  lambda={lam:.1f}: velocity={v:.3f} /r0.s")

    V.plot_rf_curves(rf_by_lambda,
                     savepath=B.fig("fig6_front_distance_strong.png"),
                     title="Front distance, strong model (Fig. 6)")
    V.plot_velocity(lam_grid, vel_by_model,
                    savepath=B.fig("fig7_velocity.png"))

    np.savez(B.data("propagation.npz"),
             lam_grid=np.array(lam_grid),
             strong_vel=np.array(vel_by_model["strong"]),
             hub_vel=np.array(vel_by_model["hub"]),
             **{f"rf_{lam}": rf_by_lambda[lam] for lam in lam_grid})
    print("\nDone.")


if __name__ == "__main__":
    main()
