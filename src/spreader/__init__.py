"""Reimplementation of Fujie & Odagaki, *Effects of superspreaders in spread of
epidemic*, Physica A 374 (2007) 843-852.

Spatial SIR Monte-Carlo on an L x L continuous torus with two superspreader
models (strong-infectiousness and hub).
"""

from .models import (
    CUTOFF_R0,
    BOX_L,
    W0,
    GAMMA,
    model_params,
    density_to_N,
    N_to_density,
    R0_analytic,
    critical_density_curve,
    RC_STRONG,
    RC_HUB,
)
from .simulator import run_single
from .runner import run_batch, single_full_run, percolation_probability

__all__ = [
    "CUTOFF_R0",
    "BOX_L",
    "W0",
    "GAMMA",
    "model_params",
    "density_to_N",
    "N_to_density",
    "R0_analytic",
    "critical_density_curve",
    "RC_STRONG",
    "RC_HUB",
    "run_single",
    "run_batch",
    "single_full_run",
    "percolation_probability",
]
