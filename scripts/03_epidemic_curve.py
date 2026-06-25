"""Fig. 8: epidemic curves (newly infected per timestep) for the strong and hub
models at lambda = 0.2, plus the no-superspreader baseline (lambda = 0).
rho*pi*r0^2 = 20.

Run:
    python scripts/03_epidemic_curve.py --runs 500
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import mean_epidemic_curve
from spreader import visualize as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--density", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(args.density)
    print(f"rho*pi*r0^2 = {args.density} -> N = {N}")

    res_strong = run_batch(N, "strong", args.lam, args.runs, n_jobs=args.jobs)
    res_hub = run_batch(N, "hub", args.lam, args.runs, n_jobs=args.jobs)
    res_none = run_batch(N, "none", 0.0, args.runs, n_jobs=args.jobs)

    # Paper averages over ALL runs ("averaging over 1000 Monte Carlo runs",
    # p.845), i.e. fizzled epidemics included -- not just outbreaks.
    c_strong = mean_epidemic_curve(res_strong, condition="all")
    c_hub = mean_epidemic_curve(res_hub, condition="all")
    c_none = mean_epidemic_curve(res_none, condition="all")

    curves = {
        fr"Strong ($\lambda={args.lam}$)":
            (c_strong, dict(marker="o", ms=3, ls="-", color="tab:red")),
        fr"Hub ($\lambda={args.lam}$)":
            (c_hub, dict(marker="s", ms=3, ls="-", mfc="none", color="tab:blue")),
        r"No superspreader ($\lambda=0$)":
            (c_none, dict(marker="^", ms=3, ls="-", mfc="none", color="tab:cyan")),
    }
    V.plot_epidemic_curve(curves, savepath=B.fig("fig8_epidemic_curve.png"))

    np.savez(B.data("epidemic_curve.npz"),
             strong=c_strong, hub=c_hub, none=c_none, lam=args.lam, N=N)
    print("Done.")


if __name__ == "__main__":
    main()
