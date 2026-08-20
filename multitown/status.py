"""Print compact progress for the latest MultiTown benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=None)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.root:
        root = Path(args.root)
    else:
        latest_advanced = project / "artifacts" / "latest-advanced-run.txt"
        latest = latest_advanced if latest_advanced.exists() else project / "artifacts" / "latest-run.txt"
        if not latest.exists():
            raise SystemExit("No benchmark run has been started.")
        root = Path(latest.read_text(encoding="utf-8").strip())
    print(f"run_root={root}")
    present = [
        child.name for child in root.iterdir()
        if child.is_dir() and child.name.startswith("A") and child.name[1:].isdigit()
    ] if root.exists() else []
    architectures = sorted(present, key=lambda value: int(value[1:]))
    if not architectures:
        architectures = ["A3", "A4", "A5"] if "advanced" in root.name else ["A0", "A1", "A2"]
    for architecture in architectures:
        status_path = root / architecture / "status.json"
        if not status_path.exists():
            print(f"{architecture}: pending")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        print(
            f"{architecture}: state={status.get('state')} "
            f"elapsed={float(status.get('elapsed_s', 0))/3600:.3f}h "
            f"remaining={float(status.get('remaining_s', 0))/3600:.3f}h "
            f"n={status.get('decisions', 0)} accuracy={float(status.get('accuracy', 0)):.4f} "
            f"tokens={status.get('total_tokens', 0)} errors={status.get('request_errors', 0)}"
        )


if __name__ == "__main__":
    main()
