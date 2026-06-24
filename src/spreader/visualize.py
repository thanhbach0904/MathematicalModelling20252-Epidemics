"""Matplotlib figures reproducing the plots of the paper.

Every function takes already-computed data and returns a Matplotlib Figure, so
the experiment scripts stay free of plotting details.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")           # headless: write PNGs without a display
import matplotlib.pyplot as plt

from .models import BOX_L


# colour cycle roughly matching the paper's lambda series
_LAMBDA_COLORS = {
    0.0: "tab:red", 0.2: "tab:green", 0.4: "tab:blue",
    0.6: "tab:purple", 0.8: "tab:cyan", 1.0: "gold",
}


def _color(lam):
    return _LAMBDA_COLORS.get(round(lam, 2), None)


def plot_w_of_r(r0=1.0, savepath=None):
    """Figs. 1-2: distance dependence of w(r) for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    r = np.linspace(0, np.sqrt(6) * r0, 400)

    # strong model
    wn = np.where(r <= r0, (1 - r / r0) ** 2, 0.0)
    ws = np.where(r <= r0, 1.0, 0.0)
    axes[0].plot(r / r0, wn, "--", label="normal", color="tab:cyan")
    axes[0].plot(r / r0, ws, "-", label="superspreader", color="tab:orange")
    axes[0].set_title("Strong infectiousness model (Fig. 1)")

    # hub model
    rs = np.sqrt(6) * r0
    wn2 = np.where(r <= r0, (1 - r / r0) ** 2, 0.0)
    ws2 = np.where(r <= rs, (1 - r / rs) ** 2, 0.0)
    axes[1].plot(r / r0, wn2, "--", label="normal", color="tab:cyan")
    axes[1].plot(r / r0, ws2, "-", label="superspreader", color="tab:orange")
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
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lam, probs in sorted(prob_by_lambda.items()):
        ax.plot(X_grid, probs, "o-", ms=3, color=_color(lam),
                label=fr"$\lambda={lam:.1f}$")
    ax.set_xlabel(r"$\rho \pi r_0^2$")
    ax.set_ylabel("percolation probability")
    ax.set_title(f"Percolation probability ({model} model)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title=None, fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_critical_density(crit_by_model, lambda_curve, curves, savepath=None):
    """Fig. 5: critical reduced density vs lambda, simulation + R0=Rc curves.

    crit_by_model : {model: (lam_array, Xc_array)} simulation points
    curves        : {model: Xc_array}  analytic R0=Rc curve on ``lambda_curve``
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    styles = {"strong": ("o", "tab:red", "-", "Strong infectiousness"),
              "hub": ("s", "tab:blue", "--", "Hub")}
    for model, (lam, Xc) in crit_by_model.items():
        m, c, ls, name = styles[model]
        ax.plot(lam, Xc, m, color=c, label=f"{name} (simulation)")
        ax.plot(lambda_curve, curves[model], ls, color=c,
                label=f"{name} ($R_0=R_c$)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\rho_c \pi r_0^2$")
    ax.set_title("Critical density vs superspreader fraction (Fig. 5)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_rf_curves(rf_by_lambda, savepath=None, title="Front distance (Fig. 6)"):
    """Fig. 6: r_f / r0 vs time step for several lambda."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lam, rf in sorted(rf_by_lambda.items()):
        t = np.arange(len(rf))
        ax.plot(t, rf, "-", color=_color(lam), label=fr"$\lambda={lam:.1f}$")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$r_f / r_0$")
    ax.set_xlim(0, min(40, len(next(iter(rf_by_lambda.values())))))
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_velocity(lam_grid, vel_by_model, savepath=None):
    """Fig. 7: propagation velocity vs lambda for both models."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"strong": ("o", "tab:red", "Strong infectiousness model"),
               "hub": ("s", "tab:blue", "Hub model")}
    for model, vel in vel_by_model.items():
        m, c, name = markers[model]
        ax.plot(lam_grid, vel, m + "-", color=c, label=name)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"velocity $(/r_0\cdot s)$")
    ax.set_title("Propagation velocity (Fig. 7)")
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_epidemic_curve(curves, savepath=None, title="Epidemic curves (Fig. 8)"):
    """Fig. 8: newly infected per timestep for several model/lambda labels.

    ``curves`` : {label: (curve_array, style_dict)}
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, (curve, style) in curves.items():
        t = np.arange(len(curve))
        ax.plot(t, curve, label=label, **style)
    ax.set_xlabel("time step")
    ax.set_ylabel("number of newly infected")
    ax.set_xlim(0, 40)
    ax.set_title(title)
    ax.legend(fontsize=8)
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

    fig, ax = plt.subplots(figsize=(6, 6))

    # draw infection arrows where the minimum-image edge does not cross a
    # periodic boundary (keeps the picture readable, as in the paper)
    for j in range(len(infector)):
        i = infector[j]
        if i < 0:
            continue
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if abs(dx) < 0.5 * L and abs(dy) < 0.5 * L:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    "-", color="0.4", lw=0.5, zorder=1)

    infected = state != 0
    # susceptible
    sN = (~infected) & (is_super == 0)
    sS = (~infected) & (is_super != 0)
    # infected (ever)
    iN = infected & (is_super == 0)
    iS = infected & (is_super != 0)

    ax.scatter(pos[sN, 0], pos[sN, 1], s=14, facecolors="none",
               edgecolors="0.6", linewidths=0.6, label="S (normal)", zorder=2)
    ax.scatter(pos[sS, 0], pos[sS, 1], s=22, c="k", marker="o",
               label="S (superspreader)", zorder=2)
    ax.scatter(pos[iN, 0], pos[iN, 1], s=16, facecolors="none",
               edgecolors="tab:blue", linewidths=0.8, label="I (normal)", zorder=3)
    ax.scatter(pos[iS, 0], pos[iS, 1], s=28, c="tab:blue", marker="o",
               label="I (superspreader)", zorder=3)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_secondary_distribution(dists, savepath=None,
                                title="Secondary-infection distribution"):
    """Figs. 12-13: distribution of the number of links (out-degree).

    ``dists`` : {label: (centres, freq, style_dict)}
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
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
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(centres, freq, width=0.9, color="magenta")
    ax.set_xlabel("number of direct secondary cases")
    ax.set_ylabel("frequency")
    ax.set_title("SARS Singapore secondary cases (~Fig. 14, digitised)")
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_sars_comparison(model_curves, sars_curve, savepath=None):
    """Fig. 15: SARS epidemic curve vs model epidemic curves."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    t_sars = np.arange(len(sars_curve))
    ax.bar(t_sars, sars_curve, width=0.9, color="orange", alpha=0.8,
           label="data of SARS in Singapore")
    for label, (curve, style) in model_curves.items():
        t = np.arange(len(curve))
        ax.plot(t, curve, label=label, **style)
    ax.set_xlabel("time step (1 step = 6 days)")
    ax.set_ylabel("number of patients")
    ax.set_xlim(0, 25)
    ax.set_title("SARS comparison (Fig. 15)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def plot_finite_size(curves, paper_Rc, savepath=None):
    """Finite-size convergence of the critical density X_c(lambda=1) vs box size L.

    ``curves``   : {model: (L_array, Xc_array)} measured critical densities
    ``paper_Rc`` : {model: Rc} the paper's Rc reference lines (Eqs. 4-5)
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {"strong": ("o", "tab:red", "Strong infectiousness"),
              "hub": ("s", "tab:blue", "Hub")}
    for model, (Ls, Xcs) in curves.items():
        m, c, name = styles[model]
        ax.plot(Ls, Xcs, m + "-", color=c, label=f"{name} (simulated $X_c$)")
        ax.axhline(paper_Rc[model], ls="--", color=c, lw=1,
                   label=f"{name} paper $R_c={paper_Rc[model]:.1f}$")
    ax.set_xlabel(r"box size $L / r_0$")
    ax.set_ylabel(r"critical density $\rho_c \pi r_0^2$  ($\lambda=1$)")
    ax.set_title("Finite-size convergence toward the paper's $R_c$")
    ax.legend(fontsize=8)
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
                ax.set_title(model)
            if col == 0:
                ax.set_ylabel(fr"$\gamma={gamma}$")
    fig.suptitle("Epidemic curves across recovery rate gamma (Fig. S1)")
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


def plot_rn_sweep(ratios, mse_normalized, mse_unnormalized, strong_mse, lam,
                  crossing_normalized=None, crossing_unnormalized=None, savepath=None):
    """FIG_S3: hub-model MSE vs r_n/r0, normalized (solid) vs unnormalized
    (dashed), against the fixed-reference strong-model MSE (horizontal line).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ratios, mse_normalized, "s-", color="tab:blue", label="Hub (normalized)")
    ax.plot(ratios, mse_unnormalized, "s--", color="tab:purple", label="Hub (unnormalized)")
    ax.axhline(strong_mse, ls="--", color="tab:red", label="Strong (reference)")
    for crossing, c in ((crossing_normalized, "tab:blue"),
                        (crossing_unnormalized, "tab:purple")):
        if crossing is not None and np.isfinite(crossing):
            ax.axvline(crossing, ls=":", color=c)
    ax.axvline(np.sqrt(6.0), ls=":", color="0.6", lw=1, label=r"paper $r_n=\sqrt{6}r_0$")
    ax.set_xlabel(r"$r_n / r_0$")
    ax.set_ylabel("MSE vs SARS curve")
    ax.set_title(fr"MSE vs hub range ratio ($\lambda={lam}$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, savepath)
    return fig


def _save(fig, savepath):
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"  saved {savepath}")
