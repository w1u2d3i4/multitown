"""Build the immutable, stratified MASBench Stage-1 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
SOURCE_REVISION = "f1d57e20c304c5e42fa77c1a0412de2bc7ad52a3"
SUBSET_SCHEMA = "multitown-masbench-subset-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def axis_value_sort_key(value: str) -> tuple[int, tuple[Any, ...]]:
    parts = value.split(",")
    if parts and all(part.strip().isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (1, (value,))


def stable_select(
    rows: Iterable[dict[str, Any]], *, axis: str, split: str, count: int, seed: int
) -> list[dict[str, Any]]:
    """Select exactly ``count`` rows, balanced over the axis value strata."""

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["value"])].append(row)
    if not strata:
        raise ValueError(f"no rows for {axis}/{split}")

    values = sorted(strata, key=axis_value_sort_key)
    base, remainder = divmod(count, len(values))
    selected: list[dict[str, Any]] = []
    for position, value in enumerate(values):
        quota = base + (1 if position < remainder else 0)
        candidates = strata[value]
        if len(candidates) < quota:
            raise ValueError(f"{axis}/{split}/value={value}: need {quota}, have {len(candidates)}")
        ranked = sorted(
            candidates,
            key=lambda row: sha256_bytes(
                f"{seed}:{axis}:{split}:{value}:{row['source_row']}".encode("utf-8")
            ),
        )
        selected.extend(ranked[:quota])

    return sorted(selected, key=lambda row: (axis_value_sort_key(str(row["value"])), row["source_row"]))


def normalize_source_row(raw: dict[str, Any], *, axis: str, split: str, source_row: int) -> dict[str, Any]:
    prompt = json.loads(raw["prompt_json"])
    reward = json.loads(raw["reward_model_json"])
    extra = json.loads(raw["extra_info_json"])
    if raw["axis"] != axis:
        raise ValueError(f"axis mismatch at {axis}/{split}/{source_row}: {raw['axis']!r}")
    if not isinstance(prompt, list) or not prompt or not all("role" in item and "content" in item for item in prompt):
        raise ValueError(f"invalid prompt at {axis}/{split}/{source_row}")
    ground_truth = str(reward["ground_truth"])
    answers = ground_truth.split("<<horizon>>")
    if not answers or any(not answer.strip() for answer in answers):
        raise ValueError(f"invalid ground truth at {axis}/{split}/{source_row}")
    record = {
        "schema_version": SUBSET_SCHEMA,
        "sample_id": f"masbench:{axis}:{split}:{source_row}",
        "axis": axis,
        "axis_value": str(raw["value"]),
        "split": split,
        "source_row": source_row,
        "messages": prompt,
        "answers": answers,
        "extra_info": extra,
    }
    record["sample_sha256"] = sha256_bytes(canonical_json(record).encode("utf-8"))
    return record


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised by reproduction environment
        raise RuntimeError("MASBench preparation requires pyarrow; use the multitown-uno or hello-agents environment") from exc
    table = parquet.read_table(path)
    return table.to_pylist()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_root: Path, output_root: Path, *, per_axis_split: int, seed: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for axis in AXES:
        for split in ("train", "test"):
            matches = sorted((source_root / axis).glob(f"{split}-*.parquet"))
            if len(matches) != 1:
                raise ValueError(f"expected one parquet file for {axis}/{split}, found {len(matches)}")
            source_path = matches[0]
            raw_rows = _read_parquet(source_path)
            rows = [
                {
                    **dict(raw),
                    "source_row": index,
                }
                for index, raw in enumerate(raw_rows)
            ]
            selected = stable_select(rows, axis=axis, split=split, count=per_axis_split, seed=seed)
            all_records.extend(
                normalize_source_row(row, axis=axis, split=split, source_row=row["source_row"])
                for row in selected
            )
            source_files.append(
                {
                    "path": str(source_path.relative_to(source_root)),
                    "size_bytes": source_path.stat().st_size,
                    "sha256": _file_sha256(source_path),
                    "row_count": len(raw_rows),
                }
            )

    subset_path = output_root / "subset.jsonl"
    subset_bytes = b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in all_records
    )
    subset_path.write_bytes(subset_bytes)

    strata = Counter((record["axis"], record["split"], record["axis_value"]) for record in all_records)
    manifest = {
        "schema_version": "multitown-masbench-manifest-v1",
        "dataset": "Salesforce/MASBench",
        "source_revision": SOURCE_REVISION,
        "selection_algorithm": "sha256-ranked stratified sampling without replacement",
        "selection_seed": seed,
        "per_axis_split": per_axis_split,
        "sample_count": len(all_records),
        "subset_path": "subset.jsonl",
        "subset_size_bytes": len(subset_bytes),
        "subset_sha256": sha256_bytes(subset_bytes),
        "strata": [
            {"axis": axis, "split": split, "value": value, "count": count}
            for (axis, split, value), count in sorted(strata.items())
        ],
        "source_files": source_files,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-axis-split", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.per_axis_split <= 0:
        parser.error("--per-axis-split must be positive")
    manifest = prepare(
        args.source_root.resolve(),
        args.output_root.resolve(),
        per_axis_split=args.per_axis_split,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
