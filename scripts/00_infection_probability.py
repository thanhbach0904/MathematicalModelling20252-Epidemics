"""Figs. 1-2: the distance dependence of the infection probability w(r) for the
strong-infectiousness and hub models.  No simulation needed.

Run:
    python scripts/00_infection_probability.py
"""

import _bootstrap as B
from spreader import visualize as V


def main():
    V.plot_w_of_r(savepath=B.fig("fig1_2_infection_probability.png"))
    print("Done.")


if __name__ == "__main__":
    main()
