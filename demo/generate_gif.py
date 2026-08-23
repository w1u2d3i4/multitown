#!/usr/bin/env python3
"""Render the deterministic MultiTown Arena replay into a README-ready GIF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "assets" / "multitown-arena.gif"
SCENE_COUNT = 7


def find_firefox() -> str:
    firefox = shutil.which("firefox")
    if firefox is None:
        raise SystemExit("Firefox is required to render the demo GIF.")
    return firefox


def screenshot(firefox: str, profile: Path, destination: Path, step: int) -> None:
    query = urlencode({"capture": "1", "step": step})
    url = f"{(ROOT / 'index.html').as_uri()}?{query}"
    command = [
        firefox,
        "--headless",
        "--no-remote",
        "--profile",
        str(profile),
        "--window-size",
        "960,540",
        "--screenshot",
        str(destination),
        url,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=45)
    if completed.returncode != 0 or not destination.exists():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Firefox failed to render scene {step}: {detail}")


def quantize(frame: Image.Image) -> Image.Image:
    rgb = frame.convert("RGB")
    return rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def render(output: Path) -> None:
    firefox = find_firefox()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="multitown-gif-") as temporary:
        temp = Path(temporary)
        profile = temp / "firefox-profile"
        profile.mkdir()
        paths: list[Path] = []
        for step in range(SCENE_COUNT):
            frame = temp / f"frame-{step:02d}.png"
            screenshot(firefox, profile, frame, step)
            paths.append(frame)

        frames = [quantize(Image.open(path)) for path in paths]
        durations = [900, 1100, 1100, 1300, 1300, 1100, 2400]
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=True,
        )

    size_mib = output.stat().st_size / 1024 / 1024
    print(f"wrote {output} ({size_mib:.2f} MiB, {SCENE_COUNT} scenes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
