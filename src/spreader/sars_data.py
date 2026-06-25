"""SARS reference data loaders for the Section 4 comparison plots.

The data live in repository-level CSV files under ``data/`` instead of being
hard-coded here. This keeps the provenance visible and makes it easy to replace
the digitised chart values with a more authoritative table later.
"""

import csv
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SARS_SECONDARY_CASES_CSV = DATA_DIR / "sars_singapore_secondary_cases.csv"
SARS_EPIDEMIC_CURVE_CSV = DATA_DIR / "sars_singapore_epidemic_curve_6day.csv"
SARS_DAYS_PER_STEP = 6


def load_secondary_cases(path=SARS_SECONDARY_CASES_CSV):
    """Return {secondary_cases: index_patients} from the reference CSV."""
    cases = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cases[int(row["secondary_cases"])] = int(row["index_patients"])
    return cases


def load_epidemic_curve(path=SARS_EPIDEMIC_CURVE_CSV):
    """Return the 6-day binned SARS Singapore epidemic curve."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["time_step"]), float(row["new_patients"])))
    rows.sort(key=lambda x: x[0])
    return np.array([value for _, value in rows], dtype=float)


SARS_SECONDARY_CASES = load_secondary_cases()
SARS_EPIDEMIC_CURVE = load_epidemic_curve()


def sars_secondary_distribution(max_links=40, normalise=True, cases=None):
    """Return (centres, freq) histogram matching Fig. 14."""
    if cases is None:
        cases = SARS_SECONDARY_CASES
    centres = np.arange(0, max_links + 1)
    freq = np.zeros(max_links + 1, dtype=float)
    for k, v in cases.items():
        if k <= max_links:
            freq[k] = v
    if normalise:
        freq = freq / freq.sum()
    return centres, freq
