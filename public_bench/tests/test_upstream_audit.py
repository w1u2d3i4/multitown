from __future__ import annotations

import json
from pathlib import Path

import pytest

from general_mas_bench.upstream_audit import audit


def _write_lock(root: Path) -> None:
    (root / "upstream-lock.json").write_text(
        json.dumps({
            "upstreams": {
                "TeamBench": {},
                "MultiAgentBench_MARBLE": {},
            }
        }),
        encoding="utf-8",
    )


def test_audit_can_select_only_teambench(tmp_path: Path) -> None:
    _write_lock(tmp_path)
    result = audit(tmp_path, ("TeamBench",))
    assert result["selected_upstreams"] == ["TeamBench"]
    assert result["issues"] == [
        f"TeamBench: missing checkout {tmp_path / 'third_party' / 'TeamBench'}"
    ]
    assert not any("MARBLE" in issue for issue in result["issues"])


def test_audit_rejects_unknown_upstream(tmp_path: Path) -> None:
    _write_lock(tmp_path)
    with pytest.raises(ValueError, match="unknown upstreams"):
        audit(tmp_path, ("unknown",))
