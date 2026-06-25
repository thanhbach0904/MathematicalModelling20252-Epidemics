# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch reimplementation of Fujie & Odagaki, *Effects of superspreaders in the spread of epidemic*, Physica A 374 (2007) 843-852. It is a spatial SIR Monte-Carlo model on an `L x L` continuous torus (`L = 10 r0`) comparing two superspreader mechanisms — **strong infectiousness** (`a=0`, constant `w(r)`) and **hub** (`a=2`, enlarged `√6 r0` cutoff). The `√6` normalisation makes a superspreader cause the same infections per unit time in both models. Every script reproduces specific figures from the paper.

## Commands

```powershell
# Setup (Python 3.10–3.13, Numba ≥ 0.61 for 3.13)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# Tests (analytic checks + simulation smoke test)
python -m pytest tests/ -q
python -m pytest tests/test_simulator.py::test_R0_closed_form -q   # single test
python tests/test_simulator.py                                     # no-pytest runner

# Run all experiments (quick / publication statistics)
python scripts/run_all.py            # --runs 150 default, ~minutes
python scripts/run_all.py --runs 1000

# Individual experiments (each maps to figures)
python scripts/00_infection_probability.py        # Figs 1-2
python scripts/01_phase_diagram.py  --runs 1000   # Figs 3-5
python scripts/02_propagation_speed.py --runs 300 # Figs 6-7
python scripts/03_epidemic_curve.py --runs 500    # Fig  8
python scripts/04_secondary_dist.py --runs 500    # Figs 9-13
python scripts/05_sars_comparison.py --runs 500   # Figs 14-15
python scripts/06_finite_size.py --runs 300       # finite-size: Xc -> Rc vs box L
python scripts/demo_live.py --model hub --lam 0.4 # live animated GIF of one epidemic

# Regenerate Fig. 5 in a larger box to shrink finite-size bias:
python scripts/01_phase_diagram.py --L 30 --runs 500
```

Common script flags: `--runs` (MC runs per point), `--jobs` (joblib workers, `-1` = all cores), `--L` (box side; default 10), and per-script `--lambdas` / `--xmax` etc. The first run pays a one-off Numba JIT compilation cost. Outputs go to `results/figures/*.png` and `results/data/*.npz` (both gitignored).

## Architecture

There is no install step — `src/` is not a package on `sys.path` by default. Two import mechanisms exist; **match the surrounding file**:
- **Scripts** import `_bootstrap` first (e.g. `import _bootstrap as B`). `scripts/_bootstrap.py` injects `src/` onto `sys.path` and creates `results/` dirs. Use `B.fig(name)` / `B.data(name)` for output paths.
- **Tests** insert `../src` onto `sys.path` manually at the top of the file.

Data flow is a strict pipeline: **models → geometry/cell_list → simulator → runner → analysis → visualize**.

- `models.py` — the single source of truth for constants (`W0=1`, `GAMMA=1`, `BOX_L=10`, `RC_STRONG=4.5`, `RC_HUB=3.2`). `model_params(model)` translates a model name (`"strong"`, `"hub"`, `"none"`) into the per-individual `(cutoff, exponent)` pairs and the cell-list geometry the simulator consumes. Also holds the density↔N conversions (`X = ρπr0² = Nπ/100`) and the analytic `R0(λ) = X(1+5λ)/6` / critical-density closed forms. Change physics here, not in the simulator.
- `simulator.py` — `run_single(...)`, a single Numba `@njit` MC run. This is the hot kernel: one sweep processes the snapshot of currently-infected nodes in random order, each infecting susceptibles within its cutoff with prob `w(r)`; newly infected act only next sweep; an active node recovers with prob `γ`. Returns a tuple of arrays (epidemic curve, front distance, secondary/out-degree, infector tree, unwrapped positions, state).
- `runner.py` — wraps `run_single` for joblib parallelism. `run_batch` returns a list of result **dicts** (the canonical record shape consumed downstream); `percolation_probability` and `single_full_run` (with `full=True` for plotting trees) are the two convenience entry points. Each run gets a reproducible seed; note **two RNGs** must be seeded — NumPy's generator for positions/superspreader assignment, and Numba's internal RNG via `seed_rng()` for the kernel.
- `analysis.py` — pure aggregation of a `run_batch` result list into the paper's observables (mean epidemic curve, front velocity, secondary-link distribution, critical density). `_select` filters runs by `"all" | "percolated" | "outbreak"` and falls back to all runs if the filter is empty. `critical_density` takes `density_to_N` by dependency injection to avoid a circular import with `models`.
- `geometry.py` / `cell_list.py` — Numba helpers: minimum-image displacement on the torus, and the O(N) linked-cell neighbour list. **Invariant: `cell_size ≥ max cutoff`** so a 3×3 cell search always covers the interaction range; `n_cells ≥ 3` so the search never wraps onto its own cell. `model_params` enforces this.

## Conventions and gotchas

- **Periodic boundary / percolation.** The torus has no real edge, so the simulator carries an *unwrapped* position accumulated from min-image displacements along the infection tree. Following the paper's bottom→top definition, a run is "percolated" when the cluster's **vertical** extent (`uymax − uymin`) reaches `L`. With this criterion the simulated critical density lands on the analytic `R0=Rc` curve (Fig. 5) up to a ~10% finite-size shortfall at `λ=1` — guarded by `test_critical_density_lands_on_R0_eq_Rc_curve`. (Do **not** revert to either-axis spanning; it pulls `Xc` well below the curve.) The front distance `r_f` (Fig. 6), by contrast, is the **true geometric distance** from the seed to the furthest infected node — *not* the unwrapped path length. Path-accumulated distance random-walk-drifts over generations and inverts the λ ordering (low λ spreads slowly over many generations → spuriously large `r_f`); the geometric measure reproduces the paper's ordering (higher λ → faster, earlier plateau).
- **Bounded box vs torus (`periodic` flag).** `run_single`/`run_batch`/`single_full_run` take `periodic` (default `True` = min-image torus, used for the phase diagram / critical density). Fig. 6/7 (`scripts/02_propagation_speed.py`) default to a **bounded hard-wall box** (`periodic=False`; pass `--periodic` to opt back to the torus). On the bounded box the cell list skips out-of-range neighbour cells, displacement is the plain Euclidean difference, and `unwrapped == positions`. With the seed at the bottom-centre `(L/2, 0)` the front then reaches the paper's absolute scale — up to `√(5²+10²) ≈ 11.18 r0` to the top corners — instead of the torus cap `L/√2 ≈ 7.07`. The torus would cap and flatten the λ-ordered plateaus into one ~7 line.
- When adding a model variant, extend `model_params` (and the integrals in `R0_analytic`) — keep the simulator model-agnostic; it only reads `(cutoff_n, exp_n, cutoff_s, exp_s)`.
- **Ensemble averaging.** The paper averages every observable "over 1000 Monte Carlo runs" (p.845) — i.e. over *all* runs, fizzles included. Scripts therefore pass `condition="all"` to `mean_epidemic_curve` / `mean_rf_curve` / `front_velocity` (not `"outbreak"`/`"percolated"`). At the paper's densities (e.g. ρπr0²=20) ~95% of runs are full outbreaks, so the choice barely shifts the curves, but `"all"` is the faithful convention. Run figure scripts at `--L 10 --runs 1000` to reproduce the paper.
- **Fig. 8 magnitude.** Our epidemic curve peaks ~2× higher than the paper's small Fig. 8. This is *not* a bug: at ρπr0²=20, λ=0.2 the model has `R0 = X(1+5λ)/6 = 6.67`, giving ~96% final size (verified), so the newly-infected peak is ~20% of N. The paper's plotted curve appears to saturate lower (~67%), which cannot be reproduced without contradicting the stated dynamics (γ=1 and the `w(r)` integrals). Shape, peak-ordering (hub earliest/highest → strong → λ=0 broad/low) and timing match.
- `sars_data.py` is hand-digitised approximations of Figs 14-15, adequate only for qualitative comparison.
- Tests assert exact analytic relations (`density_to_N(15)==477`, `R0=X(1+5λ)/6`, critical-curve endpoints) plus a qualitative simulation check — keep these green when touching `models.py` or `simulator.py`.
