#!/usr/bin/env bash
set -euo pipefail

uv run neurofly-terrain-benchmark \
  --output-dir outputs/terrain_try \
  --trials 2 \
  --duration-s 0.05 \
  --no-render \
  --no-save-video

uv run neurofly-terrain-research \
  --output-dir outputs/terrain_research_try \
  --trials 4 \
  --duration-s 0.05
