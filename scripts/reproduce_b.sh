#!/usr/bin/env bash
set -euo pipefail

uv run neurofly-odor-benchmark \
  --output-dir outputs/odor_try \
  --trials 3 \
  --run-time 1.5 \
  --no-render \
  --no-save-video

uv run neurofly-odor-research \
  --output-dir outputs/odor_research_try \
  --trials 4 \
  --run-time 1.0
