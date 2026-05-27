#!/usr/bin/env bash
set -euo pipefail

uv run neurofly-env-check
uv run neurofly-odor-env-check
uv run ruff check .
uv run pytest
