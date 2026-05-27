# Odor Navigation Plan

## Goal

Build a showcase-ready odor-guided navigation project where the fly reaches an attractive odor source while avoiding aversive sources across randomized starts.

## Core Task

- One attractive source straight ahead in the arena
- Two aversive sources offset to the left and right
- Randomized spawn positions near the origin
- Randomized headings
- One steering policy based on bilateral attractive and aversive odor imbalance

## Outputs

- CSV with per-trial metrics
- Trajectory plot across trials
- Success-rate curve
- Representative video
- Short showcase report

## Metrics

- Success rate
- Final distance to target
- Time to target
- Path length
- Average speed
- Mean attractive signal
- Mean aversive signal

## Path Integration Note

This project uses `PathIntegrationController` as the simulation wrapper so stride-related observations are exported and the task can be extended toward home-vector / return-home experiments later.
