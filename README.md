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
python scripts/06_finite_size.py --runs 300     # finite-size: Xc -> Rc as box L grows
```

Common flags: `--runs` (MC runs per point), `--jobs` (parallel workers, `-1` =
all cores), `--density` (`rho*pi*r0^2`), `--lam` (superspreader fraction λ).

The first run pays a one-off Numba compilation cost (a few seconds).

### Live demo

Watch a single epidemic spread and see the paper's numbers build up step by step
(spatial front + infection tree, the epidemic curve of Fig. 8, the front distance
`r_f` of Fig. 6, and the running S/I/R counts). Writes an animated GIF to
`results/figures/`, or pass `--show` for an interactive window.

```powershell
python scripts/demo_live.py --model hub    --lam 0.4 --density 15
python scripts/demo_live.py --model strong --lam 0.2 --density 20 --seed 7
python scripts/demo_live.py --model hub    --lam 0.4 --show
```

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
scripts/          00-05 experiments + run_all + demo_live (animation)
tests/            smoke + analytic + critical-density reproduction checks
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
- **Percolation criterion.** The paper seeds at the bottom and calls a run
  percolated when the infection **reaches the top**. We accumulate an *unwrapped*
  position along the infection tree (minimum-image displacements) and flag a run
  when the cluster's **vertical extent** reaches `L` — i.e. it spans bottom→top
  (the cluster wraps around in `y`). This gives the clean sigmoidal transition of
  Figs. 3-4. The critical density is the 50%-crossing of the percolation
  probability; with this criterion the simulated `Xc` lands on the analytic
  `R0=Rc` curve of Fig. 5 to within a small finite-size shortfall (~10% at `λ=1`,
  where the critical `N≈140` is smallest), shrinking as `λ` decreases. The same
  unwrapped tracking lets the front distance `r_f` (Fig. 6) exceed `L/√2`.
- **SARS data** (`sars_data.py`). The Fig. 14 secondary-case distribution is now
  **CDC-verified** for its defining features (CDC MMWR 52:405-411): 201 probable
  cases, 162 with zero transmission, 39 transmitters, and the five super-spreaders
  at 12/21/23/23/40 — exactly the paper's "12, 21, 23, 40". Only the small middle
  bins (1-6 secondary cases) are reconstructed (they live only in the MMWR figure
  image). The Fig. 15 epidemic curve remains a digitised approximation of MMWR
  Fig. 1 (image-only daily counts), constrained to the right total and peak.
- **Finite-size / Fig. 5.** With the bottom→top criterion the simulated critical
  density sits just below the analytic `R0=Rc` curve at `L=10`. `06_finite_size.py`
  shows that for the **strong** model (where `Rc=4.5` is a continuum-percolation
  *theory* value) `Xc` climbs toward 4.5 as `L` grows (≈0.90 → 0.97 from L=10→30):
  the gap is genuine finite size. For the **hub** model `Rc=3.2` is the paper's own
  measured `Xc` at `L=10` (Eq. 5), not an infinite-system limit, so it does not
  converge upward. Regenerate Fig. 5 at a larger box with
  `01_phase_diagram.py --L 30`.

## Optional Tier-3 (GPU)

Only needed for `N ≳ 1e5` or full `(λ × ρ)` parameter sweeps. Uncomment
`cupy`/Numba-CUDA in `requirements.txt`; the natural strategy here is **one
CUDA thread per independent simulation** (embarrassingly parallel, `cuRAND`
per-thread RNG). Not required to reproduce the paper.
