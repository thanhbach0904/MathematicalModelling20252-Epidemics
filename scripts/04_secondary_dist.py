"""Figs. 9-13: infection-route networks and the distribution of the number of
links (secondary infections).  rho*pi*r0^2 = 15, lambda = 0.2.

Run:
    python scripts/04_secondary_dist.py --runs 500
"""

import argparse

import _bootstrap as B
import numpy as np

from spreader.models import density_to_N
from spreader.runner import run_batch, single_full_run
from spreader.analysis import secondary_distribution
from spreader import visualize as V


def _pick_outbreak_seed(N, model, lam, max_seed=200):
    """Find a seed that produced a sizeable outbreak (for a nice network plot)."""
    best_seed, best_size = 0, -1
    for s in range(max_seed):
        r = single_full_run(N, model, lam, seed=s)
        if r["total_infected"] > best_size:
            best_seed, best_size = s, r["total_infected"]
        if r["total_infected"] > 0.5 * N:
            return s
    return best_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--density", type=float, default=15.0)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    N = density_to_N(args.density)
    print(f"rho*pi*r0^2 = {args.density} -> N = {N}")

    # --- Figs. 9-11: single-run infection networks ---
    for model, label in (("strong", "Strong infectiousness (Fig. 9)"),
                         ("hub", "Hub (Fig. 10)")):
        s = _pick_outbreak_seed(N, model, args.lam)
        r = single_full_run(N, model, args.lam, seed=s)
        V.plot_infection_network(
            r, savepath=B.fig(f"fig9_10_network_{model}.png"),
            title=fr"{label}, $\lambda={args.lam}$, $\rho\pi r_0^2={args.density}$")
        print(f"  {model}: network seed={s}, infected={r['total_infected']}")

    s0 = _pick_outbreak_seed(N, "none", 0.0)
    r0 = single_full_run(N, "none", 0.0, seed=s0)
    V.plot_infection_network(
        r0, savepath=B.fig("fig11_network_none.png"),
        title=fr"No superspreader (Fig. 11), $\lambda=0$, $\rho\pi r_0^2={args.density}$")

    # --- Figs. 12-13: distributions of the number of links ---
    res_none = run_batch(N, "none", 0.0, args.runs, n_jobs=args.jobs)
    c0, f0 = secondary_distribution(res_none)
    V.plot_secondary_distribution(
        {r"$\lambda=0$ (no superspreader)":
            (c0, f0, dict(color="tab:cyan"))},
        savepath=B.fig("fig12_links_none.png"),
        title="Number of links, no superspreader (Fig. 12)")

    res_strong = run_batch(N, "strong", args.lam, args.runs, n_jobs=args.jobs)
    res_hub = run_batch(N, "hub", args.lam, args.runs, n_jobs=args.jobs)
    cs, fs = secondary_distribution(res_strong)
    ch, fh = secondary_distribution(res_hub)
    V.plot_secondary_distribution(
        {"Strong infectiousness model": (cs, fs, dict(color="tab:red")),
         "Hub model": (ch, fh, dict(color="tab:blue"))},
        savepath=B.fig("fig13_links_both.png"),
        title=fr"Number of links, $\lambda={args.lam}$ (Fig. 13)")

    np.savez(B.data("secondary_dist.npz"),
             centres=c0, none=f0, strong=fs, hub=fh)
    print("Done.")


if __name__ == "__main__":
    main()
