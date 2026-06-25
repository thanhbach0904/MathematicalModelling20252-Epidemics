"""Figs. 3-5: percolation probability vs reduced density, and the critical
density vs superspreader fraction lambda (with the analytic R0 = Rc curves).

Run:
    python scripts/01_phase_diagram.py                 # quick defaults
    python scripts/01_phase_diagram.py --runs 1000 --xmax 25 --nx 26
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N, critical_density_curve
from spreader.runner import percolation_probability
from spreader.analysis import critical_density
from spreader import visualize as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200,
                    help="Monte-Carlo runs per (lambda, density) point")
    ap.add_argument("--xmax", type=float, default=25.0,
                    help="max reduced density rho*pi*r0^2")
    ap.add_argument("--nx", type=int, default=26, help="number of density points")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--init-mode", default="bottom-random",
                    choices=["bottom-random", "bottom-center", "uniform"],
                    help="how to place the initial infected individual")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    X_grid = np.linspace(0.2, args.xmax, args.nx)

    for model in ("strong", "hub"):
        print(f"\n=== {model} model: percolation probability ===")
        prob_by_lambda = {}
        for lam in args.lambdas:
            probs = np.array([
                percolation_probability(density_to_N(X), model, lam,
                                        args.runs, n_jobs=args.jobs,
                                        init_mode=args.init_mode)
                for X in X_grid
            ])
            prob_by_lambda[lam] = probs
            print(f"  lambda={lam:.1f} done")
        # save raw
        np.savez(B.data(f"percolation_{model}.npz"),
                 X_grid=X_grid, lambdas=np.array(args.lambdas),
                 **{f"p_{lam}": prob_by_lambda[lam] for lam in args.lambdas})
        V.plot_percolation_prob(X_grid, prob_by_lambda, model,
                                savepath=B.fig(f"fig3_4_percolation_{model}.png"))

    # --- Fig. 5: critical density vs lambda ---
    print("\n=== critical density vs lambda (Fig. 5) ===")
    lam_points = [l for l in args.lambdas if l > 0] or [0.2, 0.4, 0.6, 0.8, 1.0]
    crit_by_model = {}
    for model in ("strong", "hub"):
        lams, Xcs = [], []
        for lam in lam_points:
            Xc, _, _ = critical_density(model, lam, X_grid, args.runs,
                                        n_jobs=args.jobs,
                                        init_mode=args.init_mode)
            if not np.isnan(Xc):
                lams.append(lam)
                Xcs.append(Xc)
                print(f"  {model} lambda={lam:.1f}: Xc={Xc:.2f}")
        crit_by_model[model] = (np.array(lams), np.array(Xcs))

    lam_curve = np.linspace(0.01, 1.0, 100)
    curves = {m: critical_density_curve(lam_curve, m) for m in ("strong", "hub")}
    np.savez(B.data("critical_density.npz"),
             lam_curve=lam_curve,
             strong_curve=curves["strong"], hub_curve=curves["hub"],
             strong_sim_lam=crit_by_model["strong"][0],
             strong_sim_Xc=crit_by_model["strong"][1],
             hub_sim_lam=crit_by_model["hub"][0],
             hub_sim_Xc=crit_by_model["hub"][1])
    V.plot_critical_density(crit_by_model, lam_curve, curves,
                            savepath=B.fig("fig5_critical_density.png"))
    print("\nDone. Figures in results/figures/, data in results/data/.")


if __name__ == "__main__":
    main()
