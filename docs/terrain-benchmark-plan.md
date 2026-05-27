# Terrain Benchmark Plan

## Goal

Build a teachable benchmark that compares `cpg`, `rule_based`, and `hybrid` locomotion controllers on `flat` and `rough` terrain using FlyGym 1.x (`flygym-gymnasium`).

## Why This Is The Best First Project

- It reuses the strongest tutorial path already present in the archived docs.
- It produces visuals quickly.
- It creates a clear scientific story: sensory feedback improves robustness on complex terrain.

## Build Sequence

1. Tutorial reproduction
   - Run the official "Interacting with NeuroMechFly" tutorial unchanged.
   - Run the CPG tutorial unchanged.
   - Run the rule-based tutorial unchanged.
   - Run the hybrid-controller tutorial unchanged.
2. Shared benchmark harness
   - Standardize episode duration.
   - Standardize terrain selection.
   - Standardize seeds.
   - Standardize failure detection.
3. Metrics extraction
   - Distance before failure
   - Average speed
   - Stumble count
   - Slip count
   - Success rate
4. Outputs
   - One CSV row per run
   - One MP4 per controller-terrain pair
   - One summary plot comparing robustness

## Minimal Implementation Shape

- `benchmarks/run_terrain_benchmark.py`
  - loads config
  - dispatches controller
  - runs episodes
  - writes CSV
- `benchmarks/controllers/`
  - `cpg.py`
  - `rule_based.py`
  - `hybrid.py`
- `benchmarks/analysis/plot_results.py`
  - reads CSV
  - produces summary figure

## Technical Notes

- Start with CPU rendering and short episodes.
- Keep one benchmark environment first; avoid adding olfaction or path integration to the initial locomotion comparison.
- Once the locomotion benchmark is stable, add a second phase that reuses the repo for path-integration or olfaction experiments.
