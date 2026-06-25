"""Matplotlib figures reproducing the plots of the paper.

Every function takes already-computed data and returns a Matplotlib Figure, so
the experiment scripts stay free of plotting details.

House style note: the layout deliberately *evokes* Fujie & Odagaki (marker
scatter, open/filled markers per lambda, the same axis quantities) but is not a
copy -- we add descriptive titles, a light dotted grid, thin connecting lines
and a modern palette so the figures are clearly our own reimplementation rather
than scans lifted from the paper.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")           # headless: write PNGs without a display
import matplotlib.pyplot as plt

from .models import BOX_L


# --- shared house style -----------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "axes.edgecolor": "0.3",
    "legend.framealpha": 0.92,
    "legend.fontsize": 9,
    "lines.markeredgewidth": 1.1,
})


# lambda series: (colour, marker, filled?) -- alternates open/filled markers
# like the paper, but with our own palette and marker choices.
_LAMBDA_SPEC = {
    0.0: ("#c0392b", "o", False),   # red,    open circle
    0.2: ("#27ae60", "o", True),    # green,  filled circle
    0.4: ("#2e6fb7", "s", False),   # blue,   open square
    0.6: ("#8e44ad", "s", True),    # purple, filled square
    0.8: ("#16a3b8", "^", False),   # cyan,   open triangle
    1.0: ("#e0a200", "^", True),    # amber,  filled triangle
}

# model series for the two-model comparison plots (Figs 5, 7, finite-size)
_MODEL_SPEC = {
    "strong": ("#c0392b", "o", "Strong infectiousness"),
    "hub":    ("#2e6fb7", "s", "Hub"),
}


def _lambda_kw(lam, line=True):
    """Plot kwargs for a lambda series (open vs filled marker like the paper)."""
    color, marker, filled = _LAMBDA_SPEC.get(round(lam, 2), ("0.3", "o", True))
    kw = dict(color=color, marker=marker, markersize=5.5,
              markeredgecolor=color,
              markerfacecolor=color if filled else "white",
              linewidth=1.2 if line else 0.0,
              linestyle="-" if line else "none")
    return kw


def plot_w_of_r(r0=1.0, savepath=None):
    """Figs. 1-2: distance dependence of w(r) for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    r = np.linspace(0, np.sqrt(6) * r0, 400)

    # strong model
    wn = np.where(r <= r0, (1 - r / r0) ** 2, 0.0)
    ws = np.where(r <= r0, 1.0, 0.0)
    axes[0].plot(r / r0, wn, "--", lw=1.8, label="normal", color="#16a3b8")
    axes[0].plot(r / r0, ws, "-", lw=1.8, label="superspreader", color="#e07b00")
    axes[0].set_title("Strong infectiousness model (Fig. 1)")

    # hub model
    rs = np.sqrt(6) * r0
    wn2 = np.where(r <= r0, (1 - r / r0) ** 2, 0.0)
    ws2 = np.where(r <= rs, (1 - r / rs) ** 2, 0.0)
    axes[1].plot(r / r0, wn2, "--", lw=1.8, label="normal", color="#16a3b8")
    axes[1].plot(r / r0, ws2, "-", lw=1.8, label="superspreader", color="#e07b00")
    axes[1].set_title("Hub model (Fig. 2)")

    for ax in axes:
        ax.set_xlabel(r"$r/r_0$")
        ax.set_ylabel(r"$w(r)/w_0$")
        ax.legend()
        ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_percolation_prob(X_grid, prob_by_lambda, model, savepath=None):
    """Figs. 3-4: percolation probability vs reduced density for each lambda."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for lam, probs in sorted(prob_by_lambda.items()):
        ax.plot(X_grid, probs, label=fr"$\lambda={lam:.1f}$", **_lambda_kw(lam))
    ax.set_xlabel(r"$\rho \pi r_0^2$")
    ax.set_ylabel("percolation probability")
    ax.set_title(f"Percolation probability ({model} model)")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlim(0, X_grid.max())
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_critical_density(crit_by_model, lambda_curve, curves, savepath=None):
    """Fig. 5: critical reduced density vs lambda, simulation + R0=Rc curves.

    crit_by_model : {model: (lam_array, Xc_array)} simulation points
    curves        : {model: Xc_array}  analytic R0=Rc curve on ``lambda_curve``
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    lines = {"strong": "-", "hub": "--"}
    for model, (lam, Xc) in crit_by_model.items():
        c, m, name = _MODEL_SPEC[model]
        ax.plot(lam, Xc, marker=m, linestyle="none", color=c, markersize=7,
                markerfacecolor=c, markeredgecolor="0.2",
                label=f"{name} (simulation)")
        ax.plot(lambda_curve, curves[model], lines[model], color=c, lw=1.8,
                label=f"{name} ($R_0=R_c$)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\rho_c \pi r_0^2$")
    ax.set_title("Critical density vs superspreader fraction (Fig. 5)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_rf_curves(rf_by_lambda, savepath=None, title="Front distance (Fig. 6)"):
    """Fig. 6: r_f / r0 vs time step for several lambda."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    tmax = min(40, len(next(iter(rf_by_lambda.values()))))
    for lam, rf in sorted(rf_by_lambda.items()):
        t = np.arange(len(rf))
        kw = _lambda_kw(lam)
        # sparse markers so the rising curve stays readable
        ax.plot(t, rf, label=fr"$\lambda={lam:.1f}$", markevery=3, **kw)
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$r_f / r_0$")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_velocity(lam_grid, vel_by_model, savepath=None):
    """Fig. 7: propagation velocity vs lambda for both models."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for model, vel in vel_by_model.items():
        c, m, name = _MODEL_SPEC[model]
        ax.plot(lam_grid, vel, marker=m, color=c, markersize=6.5,
                markerfacecolor=c, markeredgecolor="0.2", linewidth=1.4,
                label=f"{name} model")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"velocity $(/r_0\cdot s)$")
    ax.set_title("Propagation velocity (Fig. 7)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_epidemic_curve(curves, savepath=None, title="Epidemic curves (Fig. 8)"):
    """Fig. 8: newly infected per timestep for several model/lambda labels.

    ``curves`` : {label: (curve_array, style_dict)}
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for label, (curve, style) in curves.items():
        t = np.arange(len(curve))
        ax.plot(t, curve, label=label, **style)
    ax.set_xlabel("time step")
    ax.set_ylabel("number of newly infected")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_infection_network(result, savepath=None, title="Infection network"):
    """Figs. 9-11: spatial infection tree (uses wrapped positions + tree)."""
    pos = result["positions"]
    infector = result["infector"]
    is_super = result["is_super"]
    state = result["state"]
    L = BOX_L

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # draw the directed route of infection (infector -> infected) as arrows,
    # as in the paper. Skip edges whose minimum-image link crosses a periodic
    # boundary (keeps the picture readable).
    xs, ys, us, vs = [], [], [], []
    for j in range(len(infector)):
        i = infector[j]
        if i < 0:
            continue
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if abs(dx) < 0.5 * L and abs(dy) < 0.5 * L:
            xs.append(pos[i, 0]); ys.append(pos[i, 1])
            us.append(dx); vs.append(dy)
    if xs:
        ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1,
                  color="0.45", width=0.0028, headwidth=4.5, headlength=6,
                  headaxislength=5, alpha=0.85, zorder=1)

    infected = state != 0
    # susceptible
    sN = (~infected) & (is_super == 0)
    sS = (~infected) & (is_super != 0)
    # infected (ever)
    iN = infected & (is_super == 0)
    iS = infected & (is_super != 0)

    ax.scatter(pos[sN, 0], pos[sN, 1], s=14, facecolors="none",
               edgecolors="0.6", linewidths=0.6, label="S (normal)", zorder=2)
    ax.scatter(pos[sS, 0], pos[sS, 1], s=24, c="k", marker="o",
               label="S (superspreader)", zorder=2)
    ax.scatter(pos[iN, 0], pos[iN, 1], s=16, facecolors="none",
               edgecolors="#2e6fb7", linewidths=0.8, label="I (normal)", zorder=3)
    ax.scatter(pos[iS, 0], pos[iS, 1], s=30, c="#c0392b", marker="o",
               label="I (superspreader)", zorder=3)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(title)
    # legend: "route of infection ->" arrow first (as in the paper), then markers
    from matplotlib.lines import Line2D
    route = Line2D([0], [0], color="0.45", lw=1.0, marker=">", markersize=6,
                   markevery=[-1], label="route of infection")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([route, *handles], ["route of infection", *labels],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_secondary_distribution(dists, savepath=None,
                                title="Secondary-infection distribution"):
    """Figs. 12-13: distribution of the number of links (out-degree).

    ``dists`` : {label: (centres, freq, style_dict)}
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    labels = list(dists.keys())
    width = 0.8 / max(1, len(labels))
    for k, (label, (centres, freq, style)) in enumerate(dists.items()):
        offset = (k - (len(labels) - 1) / 2) * width
        ax.bar(np.asarray(centres) + offset, freq, width=width,
               label=label, **style)
    ax.set_xlabel("the number of links")
    ax.set_ylabel("frequency")
    ax.set_xlim(-0.5, 20)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_sars_secondary(centres, freq, savepath=None):
    """Fig. 14-style: SARS Singapore direct-secondary-case distribution."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(centres, freq, width=0.9, color="#b03a8c")
    ax.set_xlabel("number of direct secondary cases")
    ax.set_ylabel("number")
    ax.set_title("SARS Singapore secondary cases (~Fig. 14, digitised)")
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_sars_comparison(model_curves, sars_curve, savepath=None):
    """Fig. 15: SARS epidemic curve vs model epidemic curves."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    t_sars = np.arange(len(sars_curve))
    ax.bar(t_sars, sars_curve, width=0.9, color="#e08a1e", alpha=0.8,
           label="data of SARS in Singapore")
    for label, (curve, style) in model_curves.items():
        t = np.arange(len(curve))
        ax.plot(t, curve, label=label, **style)
    ax.set_xlabel("time step (1 step = 6 days)")
    ax.set_ylabel("number of patients")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, None)
    ax.set_title("SARS comparison")
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_finite_size(curves, paper_Rc, savepath=None):
    """Finite-size convergence of the critical density X_c(lambda=1) vs box size L.

    ``curves``   : {model: (L_array, Xc_array)} measured critical densities
    ``paper_Rc`` : {model: Rc} the paper's Rc reference lines (Eqs. 4-5)
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for model, (Ls, Xcs) in curves.items():
        c, m, name = _MODEL_SPEC[model]
        ax.plot(Ls, Xcs, marker=m, color=c, markersize=6.5, linewidth=1.4,
                markerfacecolor=c, markeredgecolor="0.2",
                label=f"{name} (simulated $X_c$)")
        ax.axhline(paper_Rc[model], ls="--", color=c, lw=1.2,
                   label=f"{name} paper $R_c={paper_Rc[model]:.1f}$")
    ax.set_xlabel(r"box size $L / r_0$")
    ax.set_ylabel(r"critical density $\rho_c \pi r_0^2$  ($\lambda=1$)")
    ax.set_title("Finite-size convergence toward the paper's $R_c$")
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_gamma_epidemic_curves(curves, gammas, savepath=None):
    """FIG_S1: epidemic curves for each gamma value, strong vs hub side by side.

    ``curves`` : {(model, gamma): curve_array}
    """
    fig, axes = plt.subplots(len(gammas), 2, figsize=(9, 2.2 * len(gammas)),
                             sharex=True, sharey=True)
    model_col = {"strong": 0, "hub": 1}
    for row, gamma in enumerate(gammas):
        for model, col in model_col.items():
            ax = axes[row, col]
            curve = curves.get((model, gamma))
            if curve is not None:
                t = np.arange(len(curve))
                ax.plot(t, curve, color="tab:red" if model == "strong" else "tab:blue")
            ax.set_xlim(0, 40)
            if row == 0:
                ax.set_title("strong infectiousness" if model == "strong" else model)
            if col == 0:
                ax.set_ylabel(fr"$\gamma={gamma}$")
    fig.suptitle("Epidemic curves across recovery probability $\\gamma$",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_mse_curve(x_by_model, mse_by_model, xlabel, title, baseline=None,
                   baseline_label=None, vline=None, crossing=None, savepath=None):
    """Generic MSE-vs-swept-parameter plot, used for FIG_S2/S3/S4.

    ``x_by_model``/``mse_by_model`` : {model: array}. ``baseline`` draws a
    horizontal reference line (e.g. the strong model's fixed MSE in the rn
    sweep). ``vline`` marks a parameter value of interest (e.g. lambda=0.025).
    ``crossing`` marks a detected crossover x-value.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {"strong": ("o-", "tab:red", "Strong infectiousness"),
              "hub": ("s-", "tab:blue", "Hub")}
    for model, x in x_by_model.items():
        ls, c, name = styles.get(model, ("o-", "tab:gray", model))
        ax.plot(x, mse_by_model[model], ls, color=c, label=name)
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="0.4",
                   label=baseline_label or "reference")
    if vline is not None:
        ax.axvline(vline, ls=":", color="0.4", label=f"x={vline}")
    if crossing is not None and np.isfinite(crossing):
        ax.axvline(crossing, ls="-.", color="green",
                   label=f"crossover x={crossing:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE vs SARS curve")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def _save(fig, savepath):
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"  saved {savepath}")
