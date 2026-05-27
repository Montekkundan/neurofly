from __future__ import annotations

import importlib
import platform
from importlib.metadata import version


def main() -> int:
    print("Neurofly Odor environment check")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    try:
        importlib.import_module("flygym_gymnasium")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"Import failed: {exc}")
        return 1

    print(f"flygym_gymnasium import: ok (version={version('flygym-gymnasium')})")
    print("Target track: FlyGym 1.x odor-guided navigation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
