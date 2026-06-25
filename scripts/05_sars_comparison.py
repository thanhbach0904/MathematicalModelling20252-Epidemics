"""Figs. 14-15: comparison with SARS Singapore 2003.

The paper's best fit is the hub model with N = 477 (rho*pi*r0^2 = 15),
lambda = 0.4, gamma = 1, and 1 time step = 6 days.  The SARS reference arrays
are hand-digitised approximations (see spreader.sars_data).

Run:
    python scripts/05_sars_comparison.py --runs 500
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import mean_epidemic_curve, secondary_distribution
from spreader import visualize as V
from spreader import sars_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--density", type=float, default=15.0)
    ap.add_argument("--init-mode", default="bottom-random",
                    choices=["bottom-random", "bottom-center", "uniform"],
                    help="how to place the initial infected individual")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(args.density)
    print(f"rho*pi*r0^2 = {args.density} -> N = {N} (paper uses N=477)")

    # --- Fig. 14: SARS secondary-case distribution (reference) ---
    c_sars, f_sars = sars_data.sars_secondary_distribution()
    V.plot_sars_secondary(c_sars, f_sars,
                          savepath=B.fig("fig14_sars_secondary.png"))

    # --- Fig. 15: epidemic curves, models vs SARS data ---
    res_strong = run_batch(N, "strong", args.lam, args.runs, n_jobs=args.jobs,
                           init_mode=args.init_mode)
    res_hub = run_batch(N, "hub", args.lam, args.runs, n_jobs=args.jobs,
                        init_mode=args.init_mode)
    res_none = run_batch(N, "none", 0.0, args.runs, n_jobs=args.jobs,
                         init_mode=args.init_mode)

    c_strong = mean_epidemic_curve(res_strong, condition="outbreak")
    c_hub = mean_epidemic_curve(res_hub, condition="outbreak")
    c_none = mean_epidemic_curve(res_none, condition="outbreak")

    sars = sars_data.SARS_EPIDEMIC_CURVE
    # scale model curves so their peak matches the SARS peak (shape comparison)
    def rescale(c):
        m = c.max()
        return c * (sars.max() / m) if m > 0 else c

    model_curves = {
        fr"Strong ($\lambda={args.lam}$)":
            (rescale(c_strong), dict(marker="o", ms=3, ls="", color="tab:red")),
        fr"Hub ($\lambda={args.lam}$)":
            (rescale(c_hub), dict(marker="s", ms=3, ls="", mfc="none", color="tab:blue")),
        r"No superspreader ($\lambda=0$)":
            (rescale(c_none), dict(marker="^", ms=3, ls="", mfc="none", color="tab:cyan")),
    }
    V.plot_sars_comparison(model_curves, sars,
                           savepath=B.fig("fig15_sars_comparison.png"))

    np.savez(B.data("sars_comparison.npz"),
             sars_curve=sars, strong=c_strong, hub=c_hub, none=c_none,
             sars_centres=c_sars, sars_secondary=f_sars)
    print("Done. NOTE: SARS reference data are loaded from data/*.csv.")


if __name__ == "__main__":
    main()
