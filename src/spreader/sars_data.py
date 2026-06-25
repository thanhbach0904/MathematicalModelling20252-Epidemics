"""SARS reference data for Singapore 2003 (Section 4 comparison, Figs. 14-15).

Primary source: Centers for Disease Control and Prevention, "Severe Acute
Respiratory Syndrome -- Singapore, 2003", MMWR 52(18):405-411 (the paper's
ref. [2]).  Full text + figures:
    https://www.cdc.gov/mmwr/preview/mmwrhtml/mm5218a1.htm
    https://www.cdc.gov/mmwr/PDF/wk/mm5218.pdf

Provenance of every number below is marked [CDC-VERIFIED] (stated verbatim in
the MMWR text) or [RECONSTRUCTED] (only available inside the MMWR figure
*images*, which are not machine-readable; filled to satisfy the verified totals).
"""

import numpy as np

# ---- CDC-VERIFIED headline figures (MMWR 52:405-411, data as of Apr 30 2003) --
N_PROBABLE_CASES = 201          # [CDC-VERIFIED] total probable SARS cases
N_NO_TRANSMISSION = 162         # [CDC-VERIFIED] 81% caused no secondary cases
N_TRANSMITTERS = 39             # [CDC-VERIFIED] 201 - 162
N_DEATHS = 25                   # [CDC-VERIFIED] case-fatality rate 12.5%

# [CDC-VERIFIED] the five super-spreaders, by *probable* secondary cases each.
# These are exactly the values the paper plots in Fig. 14 ("12, 21, 23 and 40";
# note 23 occurs twice). Case 1=21, Case 2=23, Case 3=23, Case 4=40, Case 5=12.
SUPERSPREADER_SECONDARY = [12, 21, 23, 23, 40]


# --- Fig. 14: number of probable cases by their number of direct secondary
# cases.  index = number of secondary cases, value = how many such patients.
def _build_secondary_cases():
    dist = {0: N_NO_TRANSMISSION}                       # [CDC-VERIFIED]
    for k in SUPERSPREADER_SECONDARY:                   # [CDC-VERIFIED]
        dist[k] = dist.get(k, 0) + 1
    # [RECONSTRUCTED] the remaining 34 transmitters infect 1..6 people each.
    # Exact split is only in MMWR Fig. 3 (an image); this reconstruction is
    # monotone-decreasing and sums to N_TRANSMITTERS - len(superspreaders) = 34.
    middle = {1: 18, 2: 8, 3: 4, 4: 2, 5: 1, 6: 1}
    assert sum(middle.values()) == N_TRANSMITTERS - len(SUPERSPREADER_SECONDARY)
    for k, v in middle.items():
        dist[k] = dist.get(k, 0) + v
    return dist


SARS_SECONDARY_CASES = _build_secondary_cases()


def sars_secondary_distribution(max_links=40, normalise=True):
    """Return (centres, freq) histogram matching Fig. 14.

    The 0-bin and the super-spreader bins (12/21/23/23/40) are CDC-verified;
    bins 1-6 are reconstructed to match the verified totals (see module docstring).
    """
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
#
# [RECONSTRUCTED] These per-step counts are digitised from MMWR Fig. 1 (an image;
# the underlying daily counts are not published as text). The shape (single peak
# in late March) and total (~N_PROBABLE_CASES) are constrained by the CDC report,
# but individual bars carry digitisation error -- this curve is NOT exact.
# Aligned to the paper's Fig. 15: the outbreak rises from ~t=2 (the first ~2
# six-day steps are near-zero) and peaks around t=6, so the series is offset two
# steps from a naive t=0 start.
SARS_EPIDEMIC_CURVE = np.array([
    0, 0, 2, 4, 12, 20, 50, 16, 40, 27, 8, 4,
    3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)

SARS_DAYS_PER_STEP = 6
