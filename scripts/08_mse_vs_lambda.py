"""MSE of hub vs strong epidemic curve against the SARS Singapore curve, as a
function of the superspreader fraction lambda (gamma=1, rho*pi*r0^2=15).

Sweeps lambda in [0.02, 1.0] for both models and plots MSE-vs-lambda with a
vertical marker at lambda = 0.025.

Run:
    python scripts/08_mse_vs_lambda.py --runs 500
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import mean_epidemic_curve, mse_vs_sars
from spreader import visualize as V

MODELS = ["strong", "hub"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--density", type=float, default=15.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--vline", type=float, default=0.025)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    lambdas = np.array([0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 1.0])
    N = density_to_N(args.density)
    print(f"density={args.density} -> N={N}, gamma={args.gamma}, runs={args.runs}")

    mse_by_model = {}
    for model in MODELS:
        mses = []
        for lam in lambdas:
            res = run_batch(N, model, float(lam), args.runs,
                            gamma=args.gamma, n_jobs=args.jobs)
            mse = mse_vs_sars(mean_epidemic_curve(res, condition="all"))
            mses.append(mse)
            print(f"  {model} lambda={lam:.3f} -> MSE={mse:.2f}")
        mse_by_model[model] = np.array(mses)

    V.plot_mse_curve({m: lambdas for m in MODELS}, mse_by_model,
                     xlabel=r"$\lambda$",
                     title="SARS curve MSE",
                     vline=args.vline,
                     savepath=B.report_fig("fig_mse_vs_lambda.png"))
    print("Done. See report/figures/fig_mse_vs_lambda.png")


if __name__ == "__main__":
    main()
