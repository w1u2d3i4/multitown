from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def test_arena_is_self_contained_and_exposes_controls() -> None:
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    assert 'id="arena"' in html
    assert 'id="play-toggle"' in html
    assert 'data-town="a4"' in html
    assert 'data-town="a8"' in html
    assert "http://" not in html
    assert "https://" not in html


def test_arena_keeps_result_claims_and_demo_boundary_visible() -> None:
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    assert "78.89%" in html
    assert "+11.67 PP SUCCESS" in html
    assert "−76.54% TOKENS" in html
    assert "DETERMINISTIC DEMO" in html


def test_readme_gif_is_valid_and_matches_capture_viewport() -> None:
    gif = DEMO / "assets" / "multitown-arena.gif"
    assert gif.stat().st_size < 2 * 1024 * 1024
    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert image.size == (960, 540)
        assert image.n_frames == 7
