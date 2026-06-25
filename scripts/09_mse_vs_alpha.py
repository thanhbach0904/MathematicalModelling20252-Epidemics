"""MSE of hub vs strong epidemic curve against the SARS Singapore curve, as a
function of the normal-individual decay exponent alpha in {1, 2, 3}
(gamma=1, rho*pi*r0^2=15). Two panels compare lambda=0.2 and lambda=0.4 side
by side, in the same visualisation format as visualize.plot_mse_curve.

Run:
    python scripts/09_mse_vs_alpha.py --runs 500
"""

import argparse

import _bootstrap as B
import numpy as np
import matplotlib.pyplot as plt

from spreader.models import density_to_N
from spreader.runner import run_batch
from spreader.analysis import mean_epidemic_curve, mse_vs_sars
from spreader import visualize as V  # noqa: F401  (applies house-style rcParams)

MODELS = ["strong", "hub"]
ALPHAS = [1, 2, 3]
LAMBDAS = [0.2, 0.4]

# match plot_mse_curve's styling exactly
_STYLES = {"strong": ("o-", "tab:red", "Strong infectiousness"),
           "hub": ("s-", "tab:blue", "Hub")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--density", type=float, default=15.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(args.density)
    alpha_arr = np.array(ALPHAS, dtype=float)
    print(f"density={args.density} -> N={N}, gamma={args.gamma}, runs={args.runs}")

    # mse[lambda][model] -> array over ALPHAS
    mse = {lam: {m: [] for m in MODELS} for lam in LAMBDAS}
    for lam in LAMBDAS:
        for model in MODELS:
            for a in ALPHAS:
                res = run_batch(N, model, lam, args.runs, gamma=args.gamma,
                                alpha=a, n_jobs=args.jobs)
                val = mse_vs_sars(mean_epidemic_curve(res, condition="all"))
                mse[lam][model].append(val)
                print(f"  lambda={lam} {model} alpha={a} -> MSE={val:.2f}")
            mse[lam][model] = np.array(mse[lam][model])

    for lam in LAMBDAS:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for model in MODELS:
            ls, c, name = _STYLES[model]
            ax.plot(alpha_arr, mse[lam][model], ls, color=c, label=name)
        ax.axvline(2, ls=":", color="0.4", label=r"$\alpha=2$ (paper)")
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks(ALPHAS)            # integers only: 1, 2, 3
        ax.set_ylabel("MSE vs SARS curve")
        ax.set_title(fr"MSE vs SARS curve ($\lambda = {lam}$)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        tag = f"{lam}".replace(".", "p")
        out = B.report_fig(f"fig_mse_vs_alpha_lam{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
