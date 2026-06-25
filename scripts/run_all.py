"""Run every experiment end-to-end with light defaults (fast, lower statistics).
For paper-quality figures pass --runs 1000 and call the scripts individually.

Run:
    python scripts/run_all.py             # quick (~ a couple of minutes)
    python scripts/run_all.py --runs 1000 # publication statistics (slower)
"""

import argparse
import runpy
import sys

import _bootstrap as B  # noqa: F401  (sets up sys.path + results dirs)


SCRIPTS = [
    "00_infection_probability.py",
    "01_phase_diagram.py",
    "02_propagation_speed.py",
    "03_epidemic_curve.py",
    "04_secondary_dist.py",
    "05_sars_comparison.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--init-mode", default="bottom-random",
                    choices=["bottom-random", "bottom-center", "uniform"],
                    help="how to place the initial infected individual")
    args = ap.parse_args()

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name in SCRIPTS:
        print(f"\n################ {name} ################")
        argv = ["--runs", str(args.runs)] if name != "00_infection_probability.py" else []
        if name != "00_infection_probability.py":
            argv += ["--init-mode", args.init_mode]
        sys.argv = [name] + argv
        runpy.run_path(os.path.join(here, name), run_name="__main__")
    print("\nAll experiments finished. See results/figures/.")


if __name__ == "__main__":
    main()
