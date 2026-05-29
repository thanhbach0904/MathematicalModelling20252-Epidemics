"""Shared setup for the experiment scripts: make ``spreader`` importable and
create the results directories.  Import this first in every script.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
DATA_DIR = os.path.join(RESULTS, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fig(name):
    return os.path.join(FIG_DIR, name)


def data(name):
    return os.path.join(DATA_DIR, name)
