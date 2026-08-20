"""Finalize an orphaned A24 attempt as permanently invalidated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .a24_artifact_state import finalize_orphaned_attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    result = finalize_orphaned_attempt(arguments.root, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
