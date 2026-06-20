"""Live demo: watch a single epidemic spread and the paper's numbers build up.

This animates ONE Monte-Carlo run on the L x L torus and, frame by frame,
recreates the quantities the paper measures:

  * the spatial infection front (who is S / I / R, plus the infection tree),
  * newly infected per step  -> the epidemic curve (Fig. 8),
  * the front distance r_f/r0 (Fig. 6),
  * cumulative infected and the analytic R0 (Eq. 3).

Because gamma = 1 the dynamics are *generational*, so a node's depth in the
infection tree equals the time step at which it was infected. We exploit that to
reconstruct the whole time evolution from a single full run -- no change to the
(JIT-compiled) simulator needed.

Run (writes a GIF to results/figures/):
    python scripts/demo_live.py --model hub --lam 0.4 --density 15
    python scripts/demo_live.py --model strong --lam 0.2 --density 20 --seed 7
    python scripts/demo_live.py --model hub --lam 0.4 --show     # live window
"""

import argparse

import _bootstrap as B
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from spreader.models import density_to_N, N_to_density, R0_analytic, BOX_L
from spreader.runner import single_full_run


def infection_depths(infector, max_steps):
    """Generation depth of every node in the infection tree (root = seed, 0).

    Under gamma = 1 the depth equals the time step at which the node was
    infected, so this gives the full time evolution from one run.
    """
    N = infector.shape[0]
    depth = np.full(N, -1, dtype=int)
    depth[0] = 0
    for _ in range(max_steps + 1):
        parent_known = (depth == -1) & (infector >= 0)
        if not parent_known.any():
            break
        pdepth = depth[np.where(infector >= 0, infector, 0)]
        ready = parent_known & (pdepth >= 0)
        if not ready.any():
            break
        depth[ready] = pdepth[ready] + 1
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["strong", "hub", "none"], default="hub")
    ap.add_argument("--lam", type=float, default=0.4, help="superspreader fraction")
    ap.add_argument("--density", type=float, default=15.0, help="rho*pi*r0^2")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed; default = pick a seed with a sizeable outbreak")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window instead of writing a GIF")
    args = ap.parse_args()

    if args.show:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    N = density_to_N(args.density)
    lam = 0.0 if args.model == "none" else args.lam
    X = N_to_density(N)
    R0 = R0_analytic(lam, args.model, N=N)
    print(f"model={args.model}  lambda={lam}  rho*pi*r0^2={X:.2f}  N={N}  R0={R0:.2f}")

    # pick a seed that actually produces an outbreak (nicer to watch)
    if args.seed is None:
        best_seed, best_size = 0, -1
        for s in range(60):
            r = single_full_run(N, args.model, lam, seed=s)
            if r["total_infected"] > best_size:
                best_seed, best_size = s, r["total_infected"]
            if r["total_infected"] > 0.4 * N:
                break
        seed = best_seed
    else:
        seed = args.seed
    res = single_full_run(N, args.model, lam, seed=seed)

    pos = res["positions"]
    unwrapped = res["unwrapped"]
    infector = res["infector"]
    is_super = res["is_super"]
    depth = infection_depths(infector, res["n_steps"])

    ever_infected = depth >= 0
    T = int(depth[ever_infected].max())
    dist0 = np.hypot(unwrapped[:, 0] - unwrapped[0, 0],
                     unwrapped[:, 1] - unwrapped[0, 1])

    # live per-step quantities reconstructed from the tree (== analysis module)
    new_per_step = np.array([(depth == t).sum() for t in range(T + 1)])  # t=0 -> seed
    rf_per_step = np.array([dist0[(depth >= 0) & (depth <= t)].max()
                            for t in range(T + 1)])
    L = BOX_L

    # precompute drawable infection edges (skip those wrapping the boundary)
    edges = []
    for j in range(N):
        i = infector[j]
        if i < 0:
            continue
        dx, dy = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1]
        if abs(dx) < 0.5 * L and abs(dy) < 0.5 * L:
            edges.append((i, j, depth[j]))

    print(f"seed={seed}  outbreak={int(ever_infected.sum())}/{N}  "
          f"steps={T}  percolated={res['percolated']}")
    print("\n step   new   cumulative   r_f/r0")
    cum = 0
    for t in range(T + 1):
        cum += int(new_per_step[t])
        print(f"  {t:3d}  {int(new_per_step[t]):4d}    {cum:6d}      {rf_per_step[t]:.2f}")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12, 6))
    ax_map = fig.add_axes([0.04, 0.08, 0.46, 0.84])
    ax_cur = fig.add_axes([0.58, 0.56, 0.38, 0.36])
    ax_rf = fig.add_axes([0.58, 0.10, 0.38, 0.32])

    ax_map.set_xlim(0, L); ax_map.set_ylim(0, L); ax_map.set_aspect("equal")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.axhline(L, color="0.7", lw=1)  # the "top" the infection races toward
    ax_map.axhline(0, color="0.7", lw=1)

    ax_cur.set_xlim(0, T); ax_cur.set_ylim(0, max(1, new_per_step[1:].max()) * 1.15
                                           if T >= 1 else 1)
    ax_cur.set_xlabel("time step"); ax_cur.set_ylabel("newly infected")
    ax_cur.set_title("Epidemic curve (Fig. 8) — live", fontsize=9)
    ax_rf.set_xlim(0, T); ax_rf.set_ylim(0, rf_per_step.max() * 1.15)
    ax_rf.set_xlabel("time step"); ax_rf.set_ylabel(r"$r_f / r_0$")
    ax_rf.set_title("Front distance (Fig. 6) — live", fontsize=9)

    edge_lines = [ax_map.plot([], [], "-", color="0.55", lw=0.5, zorder=1)[0]
                  for _ in edges]
    sc_S = ax_map.scatter([], [], s=10, facecolors="none", edgecolors="0.7",
                          linewidths=0.5, zorder=2)
    sc_R = ax_map.scatter([], [], s=12, c="0.5", marker="o", zorder=3)
    sc_I = ax_map.scatter([], [], s=20, c="tab:red", marker="o", zorder=4)
    sc_Isuper = ax_map.scatter([], [], s=70, c="tab:red", marker="*",
                               edgecolors="k", linewidths=0.4, zorder=5)
    (cur_line,) = ax_cur.plot([], [], "o-", color="tab:blue", ms=3)
    (rf_line,) = ax_rf.plot([], [], "-", color="tab:green", lw=1.5)
    txt = ax_map.text(0.02, 1.02, "", transform=ax_map.transAxes, fontsize=10,
                      va="bottom", family="monospace")

    def frame(t):
        infectious = (depth == t)
        recovered = (depth >= 0) & (depth < t)
        susceptible = ~(depth >= 0) | (depth > t)

        sc_S.set_offsets(pos[susceptible])
        sc_R.set_offsets(pos[recovered] if recovered.any() else np.empty((0, 2)))
        inf_norm = infectious & (is_super == 0)
        inf_sup = infectious & (is_super != 0)
        sc_I.set_offsets(pos[inf_norm] if inf_norm.any() else np.empty((0, 2)))
        sc_Isuper.set_offsets(pos[inf_sup] if inf_sup.any() else np.empty((0, 2)))

        for line, (i, j, d) in zip(edge_lines, edges):
            if d <= t:
                line.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
            else:
                line.set_data([], [])

        ts = np.arange(t + 1)
        cur_line.set_data(ts, new_per_step[:t + 1])
        rf_line.set_data(ts, rf_per_step[:t + 1])

        n_inf = int((depth == t).sum())
        n_cum = int(((depth >= 0) & (depth <= t)).sum())
        n_rec = int(recovered.sum())
        n_sus = N - n_cum
        perc = "YES" if rf_per_step[t] >= 0 and (
            unwrapped[(depth >= 0) & (depth <= t), 1].max()
            - unwrapped[(depth >= 0) & (depth <= t), 1].min()) >= L else "no"
        txt.set_text(
            f"{args.model} model  lambda={lam}  rho*pi*r0^2={X:.1f}  R0={R0:.2f}\n"
            f"t={t:3d}   S={n_sus:4d}  I={n_inf:4d}  R={n_rec:4d}   "
            f"infected={n_cum}/{N}   r_f={rf_per_step[t]:.2f}   percolated={perc}")
        return (sc_S, sc_R, sc_I, sc_Isuper, cur_line, rf_line, txt, *edge_lines)

    # hold the last frame a moment at the end
    frames = list(range(T + 1)) + [T] * max(2, args.fps)
    anim = FuncAnimation(fig, frame, frames=frames, interval=1000 / args.fps,
                         blit=False, repeat=True)

    if args.show:
        plt.show()
    else:
        out = B.fig(f"demo_{args.model}_lam{lam}_X{X:.0f}.gif")
        anim.save(out, writer=PillowWriter(fps=args.fps))
        print(f"\nsaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
