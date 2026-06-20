"""Finite-size study: how the simulated critical density X_c(lambda=1)
approaches the paper's R_c as the box L is enlarged (N scaled to keep the
reduced density fixed).

This explains the ~10% shortfall of Fig. 5's markers in a finite L=10 box:

  * Strong model -- R_c = 4.5 is a continuum-percolation-theory (infinite-system)
    value, so X_c climbs toward 4.5 as L grows: the gap is genuine finite size.
  * Hub model -- R_c = 3.2 is the paper's *own* measured X_c at L=10, lambda=1
    (Eq. 5), NOT an infinite-system limit, so X_c does NOT converge to 3.2 by
    enlarging L; it just stabilises at its own finite-size value.

Run:
    python scripts/06_finite_size.py --runs 300
    python scripts/06_finite_size.py --runs 600 --Ls 10 15 20 30 40
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import RC_STRONG, RC_HUB, density_to_N
from spreader.analysis import critical_density
from spreader import visualize as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300,
                    help="Monte-Carlo runs per density point")
    ap.add_argument("--Ls", type=float, nargs="+", default=[10, 20, 30, 40],
                    help="box sizes L/r0 to scan")
    ap.add_argument("--xmax", type=float, default=12.0,
                    help="max reduced density to scan for the crossing")
    ap.add_argument("--nx", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    X_grid = np.linspace(0.5, args.xmax, args.nx)
    paper_Rc = {"strong": RC_STRONG, "hub": RC_HUB}
    curves = {}

    for model in ("strong", "hub"):
        print(f"\n=== {model} model: X_c(lambda=1) vs L  (paper R_c={paper_Rc[model]}) ===")
        Ls, Xcs = [], []
        for L in args.Ls:
            Xc, _, _ = critical_density(model, 1.0, X_grid, args.runs,
                                        n_jobs=args.jobs, L=L)
            if not np.isnan(Xc):
                Ls.append(L)
                Xcs.append(Xc)
                N = density_to_N(Xc, L=L)
                print(f"  L={L:5.1f}  N~{N:6d}  X_c={Xc:.2f}  "
                      f"ratio={Xc / paper_Rc[model]:.2f}")
        curves[model] = (np.array(Ls), np.array(Xcs))

    np.savez(B.data("finite_size.npz"),
             strong_L=curves["strong"][0], strong_Xc=curves["strong"][1],
             hub_L=curves["hub"][0], hub_Xc=curves["hub"][1],
             Rc_strong=RC_STRONG, Rc_hub=RC_HUB)
    V.plot_finite_size(curves, paper_Rc,
                       savepath=B.fig("fig16_finite_size.png"))
    print("\nDone. See results/figures/fig16_finite_size.png")


if __name__ == "__main__":
    main()
