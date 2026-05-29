# MathematicalModelling20252-Epidemics

Reimplementation of **R. Fujie & T. Odagaki, *Effects of superspreaders in the
spread of epidemic*, Physica A 374 (2007) 843-852.**

A spatial SIR Monte-Carlo model on an `L x L` continuous torus (`L = 10 r0`)
with two superspreader mechanisms:

| model | normal `w(r)` | superspreader `w(r)` |
|-------|---------------|----------------------|
| **strong infectiousness** | `w0 (1 - r/r0)^2`, cutoff `r0` | `w0` (constant), cutoff `r0` |
| **hub** | `w0 (1 - r/r0)^2`, cutoff `r0` | `w0 (1 - r/(√6 r0))^2`, cutoff `√6 r0` |

The `√6` factor normalises the two models so a superspreader causes the same
number of infections per unit time in both.

## Stack

Python + NumPy + **Numba** (JIT-compiled hot loops) + **joblib** (1000 MC runs
are embarrassingly parallel). A **linked-cell list** gives O(N) neighbour
lookup per sweep instead of O(N²). Visualization is Matplotlib
(`src/spreader/visualize.py`).

## Setup

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10–3.13 (Numba ≥ 0.61 for 3.13).

## Run

No config files — everything runs from the terminal with sensible defaults and
optional CLI flags. Outputs go to `results/figures/` (PNG) and
`results/data/` (`.npz`).

```powershell
# everything, quick (lower statistics, ~minutes)
python scripts/run_all.py

# paper-quality statistics
python scripts/run_all.py --runs 1000
```

Individual experiments (each maps to figures in the paper):

```powershell
python scripts/00_infection_probability.py      # Figs 1-2  w(r)
python scripts/01_phase_diagram.py  --runs 1000 # Figs 3-5  percolation + critical density
python scripts/02_propagation_speed.py --runs 300   # Figs 6-7  front distance + velocity
python scripts/03_epidemic_curve.py --runs 500  # Fig  8    epidemic curve
python scripts/04_secondary_dist.py --runs 500  # Figs 9-13 networks + link distribution
python scripts/05_sars_comparison.py --runs 500 # Figs 14-15 SARS Singapore comparison
```

Common flags: `--runs` (MC runs per point), `--jobs` (parallel workers, `-1` =
all cores), `--density` (`rho*pi*r0^2`), `--lam` (superspreader fraction λ).

The first run pays a one-off Numba compilation cost (a few seconds).

## Tests

```powershell
python -m pytest tests/ -q      # or:  python tests/test_simulator.py
```

These check the density↔population mapping (`N = 477 ↔ ρπr0² = 15`), the
closed-form `R0(λ) = X(1+5λ)/6`, the critical-density curve endpoints
(`Rc = 4.5` strong, `3.2` hub), and run a small simulation smoke test.

## Layout

```
src/spreader/
  geometry.py     position sampling, periodic (min-image) displacement
  cell_list.py    O(N) linked-cell neighbour lookup
  models.py       w(r) parameters, density<->N, analytic R0 & critical curves
  simulator.py    one full MC run (Numba kernel)
  runner.py       parallel batches (joblib) + single full run
  analysis.py     percolation prob, critical density, velocity, distributions
  visualize.py    all Matplotlib figures
  sars_data.py    digitised SARS Singapore reference data (Figs 14-15)
scripts/          00-05 experiments + run_all
tests/            smoke + analytic checks
results/          generated figures and data (gitignored)
```

## Modelling choices / notes

- **Dynamics.** One MC sweep processes the snapshot of currently-infected
  individuals in random order; each tries to infect every susceptible within
  its cutoff with probability `w(r)`; newly infected act only next sweep
  ("without new infected ones"); after acting an infective recovers with
  probability `γ`. With `γ = 1` (the paper's value) this is a generational SIR.
- **Density parameter.** The control variable is `X = ρπr0²`. With `r0 = 1`,
  `L = 10`, `X = Nπ/100`; e.g. `N = 477 → X ≈ 15`.
- **Percolation criterion.** A torus has no real "top", so we accumulate an
  *unwrapped* position along the infection tree (minimum-image displacements)
  and call a run percolated when the unwrapped cluster **spans the box** — its
  extent in x or y reaches `L` (the cluster wraps around). This gives the clean
  sigmoidal transition of Figs. 3-4. The critical density is the 50%-crossing of
  the percolation probability; in a finite `L=10` box it sits modestly below the
  mean-field `R0=Rc` curve (Fig. 5), as expected. The same unwrapped tracking
  lets the front distance `r_f` (Fig. 6) exceed `L/√2`.
- **SARS data** in `sars_data.py` are hand-digitised approximations of Figs.
  14-15, sufficient for the qualitative comparison; replace with CDC MMWR
  52:405-411 / WHO curves for exact values.

## Optional Tier-3 (GPU)

Only needed for `N ≳ 1e5` or full `(λ × ρ)` parameter sweeps. Uncomment
`cupy`/Numba-CUDA in `requirements.txt`; the natural strategy here is **one
CUDA thread per independent simulation** (embarrassingly parallel, `cuRAND`
per-thread RNG). Not required to reproduce the paper.
