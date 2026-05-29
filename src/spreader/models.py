"""Model definitions, fixed constants and analytic quantities.

Infection probability (Eqs. 1-2 of the paper):

    w(r) = w0 * (1 - r/c)**a   for 0 <= r < c,   else 0

with per-individual cutoff ``c`` and exponent ``a``:

    normal              : c = r0,        a = 2
    strong superspreader: c = r0,        a = 0   -> w(r) = w0 (constant)
    hub    superspreader: c = sqrt(6) r0, a = 2

The factor sqrt(6) normalises the two superspreader models so that the
per-unit-time number of infections caused by a superspreader is identical
(both integrate to w0 * pi * r0**2).
"""

import numpy as np

# ---- fixed simulation constants (paper: w0 = 1, gamma = 1, L = 10 r0) -------
CUTOFF_R0 = 1.0          # r0
BOX_L = 10.0             # L = 10 r0
W0 = 1.0                 # peak infection probability
GAMMA = 1.0              # recovery probability per active step

HUB_FACTOR = np.sqrt(6.0)

# Critical basic reproductive numbers defined in the paper (Eqs. 4-5).
RC_STRONG = 4.5          # continuum-percolation threshold (all superspreaders)
RC_HUB = 3.2             # measured critical density at lambda = 1 (hub)


def model_params(model, r0=CUTOFF_R0, L=BOX_L):
    """Return a dict of per-model parameters consumed by the simulator.

    ``model`` is one of ``"strong"``, ``"hub"`` or ``"none"`` (the last treats
    every individual as normal, used for lambda = 0 baselines).
    """
    if model == "strong":
        cutoff_n, exp_n, cutoff_s, exp_s = r0, 2.0, r0, 0.0
    elif model == "hub":
        cutoff_n, exp_n, cutoff_s, exp_s = r0, 2.0, HUB_FACTOR * r0, 2.0
    elif model in ("none", "normal"):
        cutoff_n, exp_n, cutoff_s, exp_s = r0, 2.0, r0, 2.0
    else:
        raise ValueError(f"unknown model {model!r}")

    max_cut = max(cutoff_n, cutoff_s)
    n_cells = max(3, int(L / max_cut))      # >=3 so 3x3 search never wraps onto itself
    cell_size = L / n_cells                 # exact tiling; cell_size >= max_cut
    return {
        "cutoff_n": cutoff_n,
        "exp_n": exp_n,
        "cutoff_s": cutoff_s,
        "exp_s": exp_s,
        "n_cells": n_cells,
        "cell_size": cell_size,
        "L": L,
        "r0": r0,
    }


# ---- density <-> population conversions -------------------------------------
# The control parameter in the paper is the reduced density  X = rho * pi * r0**2.
# With rho = N / L**2 and L = 10 r0:  X = N * pi / 100.

def density_to_N(X, r0=CUTOFF_R0, L=BOX_L):
    """Population N giving reduced density X = rho*pi*r0**2."""
    rho = X / (np.pi * r0 ** 2)
    return int(round(rho * L ** 2))


def N_to_density(N, r0=CUTOFF_R0, L=BOX_L):
    """Reduced density X = rho*pi*r0**2 for a population of N."""
    rho = N / L ** 2
    return rho * np.pi * r0 ** 2


# ---- analytic basic reproductive number -------------------------------------
def _ring_integral(cutoff, exponent):
    """Integral_0^cutoff w0 (1-r/c)^a 2 pi r dr = 2 pi c^2 / ((a+1)(a+2))."""
    a = exponent
    return W0 * 2.0 * np.pi * cutoff ** 2 / ((a + 1.0) * (a + 2.0))


def R0_analytic(lam, model, r0=CUTOFF_R0, L=BOX_L, N=None, X=None):
    """Mean number of secondary infections per unit time from one infective
    (Eq. 3).  Either ``N`` or the reduced density ``X`` must be supplied.
    """
    if X is None:
        if N is None:
            raise ValueError("supply either N or X")
        X = N_to_density(N, r0, L)
    rho = X / (np.pi * r0 ** 2)

    p = model_params(model, r0, L)
    I_n = _ring_integral(p["cutoff_n"], p["exp_n"])
    I_s = _ring_integral(p["cutoff_s"], p["exp_s"])
    return rho * (lam * I_s + (1.0 - lam) * I_n) / W0


def critical_density_curve(lam, model):
    """Critical reduced density X_c(lambda) from the condition R0 = Rc.

    Closed form: with I_s = pi r0^2 and I_n = pi r0^2 / 6 (both models share
    these integrals after sqrt(6) normalisation),

        R0 = X * (1 + 5*lambda) / 6  ==> X_c = 6 Rc / (1 + 5 lambda).
    """
    Rc = RC_STRONG if model == "strong" else RC_HUB
    lam = np.asarray(lam, dtype=float)
    return 6.0 * Rc / (1.0 + 5.0 * lam)
