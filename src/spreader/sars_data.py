"""Approximate SARS reference data for Singapore 2003, used for the Section 4
comparison plots (Figs. 14-15).

NOTE: these arrays are *hand-digitised approximations* read off the figures in
Fujie & Odagaki (2007), themselves based on CDC MMWR 52:405-411 and WHO epidemic
curves.  They reproduce the qualitative shape (the long-tailed secondary-case
distribution with isolated superspreaders at 12/21/23/40, and the single-peaked
epidemic curve) and are NOT the exact published counts.  Replace with the
primary source if exact figures are required.
"""

import numpy as np

# --- Fig. 14: number of probable cases by their number of direct secondary
# cases (Feb 25 - Apr 30, 2003).  index = number of secondary cases, value =
# how many index patients.  Most patients infect nobody; four superspreaders
# infected 12, 21, 23 and 40 people.
SARS_SECONDARY_CASES = {
    0: 162, 1: 22, 2: 8, 3: 4, 4: 2, 5: 1,
    7: 1, 12: 1, 21: 1, 23: 1, 40: 1,
}


def sars_secondary_distribution(max_links=40, normalise=True):
    """Return (centres, freq) histogram matching Fig. 14."""
    centres = np.arange(0, max_links + 1)
    freq = np.zeros(max_links + 1, dtype=float)
    for k, v in SARS_SECONDARY_CASES.items():
        if k <= max_links:
            freq[k] = v
    if normalise:
        freq = freq / freq.sum()
    return centres, freq


# --- Fig. 15: epidemic curve (new probable cases per 6-day step),
# Feb 13 - Jun 13, 2003.  One time step == 6 days.
SARS_EPIDEMIC_CURVE = np.array([
    2, 4, 12, 20, 50, 16, 40, 27, 8, 4, 3, 2,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)

SARS_DAYS_PER_STEP = 6
