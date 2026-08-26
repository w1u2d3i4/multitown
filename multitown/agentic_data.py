"""Audit and freeze the external data used by the Agentic RL branch.

The datasets live outside the Git repository.  This module records their
content hashes, row counts, schemas, and the exact public benchmark slices used
by MultiTown.  It deliberately distinguishes a downloaded README from an
available gated payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "multitown-agentic-data-manifest-v1"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("MULTITOWN_AGENTIC_DATA_ROOT", "data/agentic-rl")
)


@dataclass(frozen=True)
class PayloadSpec:
    key: str
    family: str
    dataset_id: str
    revision: str | None
    paths: tuple[str, ...]
    format: str
    split: str
    expected_rows: int
    required_columns: tuple[str, ...]
    protocol: str


SPECS = (
    PayloadSpec(
        "apps_train",
        "agentconductor",
        "codeparrot/apps",
        "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        ("agentconductor/codeparrot__apps/train.jsonl",),
        "jsonl",
        "train",
        5_000,
        ("id", "question", "solutions", "input_output", "difficulty"),
        "AgentConductor source benchmark; original train split",
    ),
    PayloadSpec(
        "apps_test",
        "agentconductor",
        "codeparrot/apps",
        "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        ("agentconductor/codeparrot__apps/test.jsonl",),
        "jsonl",
        "test",
        5_000,
        ("id", "question", "solutions", "input_output", "difficulty"),
        "AgentConductor source benchmark; original test split",
    ),
    PayloadSpec(
        "humaneval_test",
        "agentconductor",
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        (
            (
                "agentconductor/openai__openai_humaneval/"
                "openai_humaneval/test-00000-of-00001.parquet"
            ),
        ),
        "parquet",
        "test",
        164,
        ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
        "AgentConductor source benchmark; original test split",
    ),
    PayloadSpec(
        "mbpp_test",
        "agentconductor",
        "google-research-datasets/mbpp",
        "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        (
            (
                "agentconductor/google-research-datasets__mbpp/"
                "full/test-00000-of-00001.parquet"
            ),
        ),
        "parquet",
        "full/test",
        500,
        ("task_id", "text", "code", "test_list"),
        "AgentConductor source benchmark; full test configuration",
    ),
    PayloadSpec(
        "livecodebench_release_v4",
        "agentconductor",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        tuple(
            "agentconductor/livecodebench__code_generation_lite/"
            + ("test.jsonl" if index == 1 else f"test{index}.jsonl")
            for index in range(1, 5)
        ),
        "jsonl",
        "release_v4/test",
        713,
        (
            "question_title",
            "question_content",
            "question_id",
            "difficulty",
            "public_test_cases",
            "private_test_cases",
        ),
        "AgentConductor paper release-v4 benchmark (v1 through v4 files)",
    ),
    PayloadSpec(
        "codecontests_train",
        "agentconductor",
        "deepmind/code_contests",
        "802411c3010cb00d1b05bad57ca77365a3c699d6",
        tuple(
            "agentconductor/deepmind__code_contests/data/"
            f"train-{index:05d}-of-00039-{suffix}.parquet"
            for index, suffix in enumerate(
                (
                    "e991a271dbfa9925",
                    "e092fe56fda18715",
                    "9cea23812e920e41",
                    "e3822fccad6e083a",
                    "cefe355b4667b27e",
                    "b7580d2d846c2136",
                    "65184bb9f7d61fde",
                    "05785de21e8b8429",
                    "7246e6b7423b404f",
                    "b8c920f6629b57b2",
                    "6de28ba20654f69b",
                    "5de236be5188959d",
                    "da9476a39a1bdbb7",
                    "30b8c3829ee3b962",
                    "dc3ebb07a3cba8e4",
                    "19ccd7331d695677",
                    "bf38b0908b322307",
                    "ae5533a2f822e6ef",
                    "8c793837880f5507",
                    "d688fad5ee604390",
                    "5d59387098675b73",
                    "b257bf03d6876780",
                    "1cfd39fa43c1917c",
                    "d078bcb55e45cbf0",
                    "f4e3da0e5661e6d1",
                    "3f6ebfbaba5f4c70",
                    "7d4898300894cbbe",
                    "f8196766547533a2",
                    "79a302af3c924863",
                    "2b6615897d038115",
                    "4135cc54050afc22",
                    "40309dd907c042b7",
                    "7b7d2068a3d9c359",
                    "53b0f749aacff9c1",
                    "a36ff0bff7d2a76f",
                    "d28f9be60314601f",
                    "146e1a11c054aeab",
                    "995207c374a4e6f2",
                    "96a59dd6a98cd075",
                )
            )
        ),
        "parquet",
        "train",
        13_328,
        ("name", "description", "public_tests", "private_tests", "solutions"),
        "AgentConductor source benchmark; original train shards",
    ),
    PayloadSpec(
        "codecontests_validation",
        "agentconductor",
        "deepmind/code_contests",
        "802411c3010cb00d1b05bad57ca77365a3c699d6",
        (
            (
                "agentconductor/deepmind__code_contests/data/"
                "valid-00000-of-00001-5e672c5751f060d3.parquet"
            ),
        ),
        "parquet",
        "validation",
        117,
        ("name", "description", "public_tests", "private_tests", "solutions"),
        "AgentConductor source benchmark; original validation split",
    ),
    PayloadSpec(
        "codecontests_test",
        "agentconductor",
        "deepmind/code_contests",
        "802411c3010cb00d1b05bad57ca77365a3c699d6",
        (
            (
                "agentconductor/deepmind__code_contests/data/"
                "test-00000-of-00001-9c49eeff30aacaa8.parquet"
            ),
        ),
        "parquet",
        "test",
        165,
        ("name", "description", "public_tests", "private_tests", "solutions"),
        "AgentConductor source benchmark; original test split",
    ),
    PayloadSpec(
        "magrpo_tldr_train",
        "magrpo",
        "trl-lib/tldr",
        "21233da376667088e6eb1ce4ce19ed832c2935d3",
        ("magrpo/trl-lib__tldr/data/train-00000-of-00001.parquet",),
        "parquet",
        "train",
        116_722,
        ("prompt", "completion"),
        "Official writing task; frozen training slice is rows [0,1000)",
    ),
    PayloadSpec(
        "magrpo_tldr_test",
        "magrpo",
        "trl-lib/tldr",
        "21233da376667088e6eb1ce4ce19ed832c2935d3",
        ("magrpo/trl-lib__tldr/data/test-00000-of-00001.parquet",),
        "parquet",
        "test",
        6_553,
        ("prompt", "completion"),
        "Official writing task; frozen evaluation slice is rows [0,1000)",
    ),
    PayloadSpec(
        "magrpo_tldr_validation",
        "magrpo",
        "trl-lib/tldr",
        "21233da376667088e6eb1ce4ce19ed832c2935d3",
        ("magrpo/trl-lib__tldr/data/validation-00000-of-00001.parquet",),
        "parquet",
        "validation",
        6_447,
        ("prompt", "completion"),
        "Downloaded upstream split; not selected by the frozen writing protocol",
    ),
    PayloadSpec(
        "magrpo_arxiv_train",
        "magrpo",
        "OpenMLRL/arXiv_abstract",
        "eac142a6550fff91bb1fb717c43ad3af3253ff48",
        ("magrpo/OpenMLRL__arXiv_abstract/data/train-00000-of-00001.parquet",),
        "parquet",
        "train",
        140_313,
        ("article_id", "abstract_text", "token_count"),
        "Official writing task; frozen training slice is rows [0,1000)",
    ),
    PayloadSpec(
        "magrpo_arxiv_validation",
        "magrpo",
        "OpenMLRL/arXiv_abstract",
        "eac142a6550fff91bb1fb717c43ad3af3253ff48",
        ("magrpo/OpenMLRL__arXiv_abstract/data/val-00000-of-00001.parquet",),
        "parquet",
        "validation",
        5_383,
        ("article_id", "abstract_text", "token_count"),
        "Official writing task; frozen evaluation slice is rows [0,1000)",
    ),
    PayloadSpec(
        "magrpo_arxiv_test",
        "magrpo",
        "OpenMLRL/arXiv_abstract",
        "eac142a6550fff91bb1fb717c43ad3af3253ff48",
        ("magrpo/OpenMLRL__arXiv_abstract/data/test-00000-of-00001.parquet",),
        "parquet",
        "test",
        5_481,
        ("article_id", "abstract_text", "token_count"),
        "Downloaded upstream split; not selected by the frozen writing protocol",
    ),
    PayloadSpec(
        "mattrl_rarebench_public_task4",
        "mattrl",
        "chenxz/RareBench",
        "6f054e04071953ef2c1779b279074245f2ab398c",
        ("mattrl/chenxz__RareBench/data.zip",),
        "rarebench_zip",
        "Task4/public",
        1_122,
        ("Phenotype", "RareDisease", "Department"),
        "Runnable public subset; not the unreleased 2,185-case MATTRL paper set",
    ),
    PayloadSpec(
        "mattrl_supergpqa_all",
        "mattrl",
        "m-a-p/SuperGPQA",
        "4430d4458112c7d4497fdcf94d7cc223313d6acf",
        ("mattrl/m-a-p__SuperGPQA/SuperGPQA-all.jsonl",),
        "jsonl",
        "all",
        26_529,
        (
            "uuid",
            "question",
            "options",
            "answer_letter",
            "discipline",
            "difficulty",
        ),
        "Freeze a deterministic 300-question stratified evaluation subset",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_metadata(paths: Sequence[Path]) -> tuple[int, set[str]]:
    rows = 0
    columns: set[str] = set()
    for path in paths:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL row is not an object: {path}")
                rows += 1
                columns.update(value)
    return rows, columns


def _parquet_metadata(paths: Sequence[Path]) -> tuple[int, set[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Parquet audit requires MultiTown's reproduction extra"
        ) from exc
    rows = 0
    columns: set[str] = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        columns.update(parquet.schema_arrow.names)
    return rows, columns


def _rarebench_metadata(paths: Sequence[Path]) -> tuple[int, set[str]]:
    rows = 0
    columns: set[str] = set()
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP CRC failure in {path}: {bad_member}")
            for name in archive.namelist():
                if not name.endswith(".jsonl"):
                    continue
                for line in archive.read(name).splitlines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    rows += 1
                    columns.update(value)
    return rows, columns


def _payload_metadata(spec: PayloadSpec, paths: Sequence[Path]) -> tuple[int, set[str]]:
    if spec.format == "jsonl":
        return _jsonl_metadata(paths)
    if spec.format == "parquet":
        return _parquet_metadata(paths)
    if spec.format == "rarebench_zip":
        return _rarebench_metadata(paths)
    raise ValueError(f"unsupported payload format: {spec.format}")


def _select_supergpqa(path: Path, count: int = 300) -> dict[str, Any]:
    """Select a deterministic approximately stratified subset without leakage."""
    candidates: list[tuple[str, str, str]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["uuid"])
            stratum = f"{row['discipline']}::{row['difficulty']}"
            rank = hashlib.sha256(
                f"multitown-supergpqa-v1::{row_id}".encode()
            ).hexdigest()
            candidates.append((stratum, rank, row_id))

    by_stratum: dict[str, list[tuple[str, str]]] = {}
    for stratum, rank, row_id in candidates:
        by_stratum.setdefault(stratum, []).append((rank, row_id))
    for rows in by_stratum.values():
        rows.sort()

    total = len(candidates)
    quotas = {key: count * len(rows) / total for key, rows in by_stratum.items()}
    allocated = {key: int(value) for key, value in quotas.items()}
    remainder = count - sum(allocated.values())
    remainder_order = sorted(
        by_stratum,
        key=lambda key: (-(quotas[key] - allocated[key]), key),
    )
    for key in remainder_order[:remainder]:
        allocated[key] += 1

    selected = sorted(
        row_id
        for key, rows in by_stratum.items()
        for _, row_id in rows[: allocated[key]]
    )
    digest = hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()
    return {
        "algorithm": "proportional discipline+difficulty, sha256 rank v1",
        "count": len(selected),
        "id_field": "uuid",
        "ids": selected,
        "ids_sha256": digest,
    }


def audit_data(data_root: Path, *, hash_files: bool = True) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in SPECS:
        paths = [data_root / relative for relative in spec.paths]
        missing = [
            relative for relative, path in zip(spec.paths, paths) if not path.is_file()
        ]
        entry: dict[str, Any] = asdict(spec)
        entry["paths"] = list(spec.paths)
        if missing:
            entry.update({"status": "missing", "missing_paths": missing})
            payloads.append(entry)
            continue
        try:
            rows, columns = _payload_metadata(spec, paths)
        except (TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            entry.update({"status": "invalid", "error": str(exc)})
            payloads.append(entry)
            continue
        missing_columns = sorted(set(spec.required_columns) - columns)
        file_records = []
        for relative, path in zip(spec.paths, paths):
            record = {"path": relative, "bytes": path.stat().st_size}
            if hash_files:
                record["sha256"] = _sha256(path)
            file_records.append(record)
        status = "ready"
        if rows != spec.expected_rows or missing_columns:
            status = "invalid"
        entry.update(
            {
                "status": status,
                "rows": rows,
                "columns": sorted(columns),
                "missing_columns": missing_columns,
                "files": file_records,
            }
        )
        payloads.append(entry)

    by_key = {entry["key"]: entry for entry in payloads}
    supergpqa_path = data_root / SPECS[-1].paths[0]
    selections: dict[str, Any] = {
        "magrpo_tldr_v1": {
            "train": {"payload": "magrpo_tldr_train", "start": 0, "stop": 1000},
            "evaluation": {
                "payload": "magrpo_tldr_test",
                "start": 0,
                "stop": 1000,
            },
        },
        "magrpo_arxiv_v1": {
            "train": {"payload": "magrpo_arxiv_train", "start": 0, "stop": 1000},
            "evaluation": {
                "payload": "magrpo_arxiv_validation",
                "start": 0,
                "stop": 1000,
            },
        },
    }
    if by_key["mattrl_supergpqa_all"]["status"] == "ready":
        selections["mattrl_supergpqa_300_v1"] = _select_supergpqa(supergpqa_path)

    family_status = {}
    for family in ("agentconductor", "magrpo", "mattrl"):
        statuses = [entry["status"] for entry in payloads if entry["family"] == family]
        family_status[family] = "ready" if set(statuses) == {"ready"} else "incomplete"
    family_status["mattrl_exact_paper_suite"] = "blocked"

    return {
        "schema": MANIFEST_SCHEMA,
        "data_root_recorded_as": "MULTITOWN_AGENTIC_DATA_ROOT",
        "status": (
            "ready_with_declared_gaps"
            if family_status["agentconductor"] == family_status["magrpo"] == "ready"
            else "incomplete"
        ),
        "families": family_status,
        "declared_gaps": [
            {
                "dataset": "cais/hle",
                "status": "gated_payload_missing",
                "reason": "dataset access terms and authenticated download are required",
            },
            {
                "dataset": "MATTRL RareBench paper set",
                "status": "exact_split_unavailable",
                "reason": (
                    "the public RareBench payload has 1,122 Task-4 cases; the "
                    "MATTRL code release does not publish its 2,185-case split"
                ),
            },
            {
                "method": "AgentConductor",
                "status": "official_training_code_unavailable",
                "reason": "paper protocol is public but no official executable repository was found",
            },
            {
                "method": "MATTRL",
                "status": "official_training_code_unavailable",
                "reason": "the official repository currently contains documentation only",
            },
        ],
        "payloads": payloads,
        "selections": selections,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-file-hashes",
        action="store_true",
        help="skip SHA-256 hashing for a faster diagnostic audit",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = audit_data(args.data_root, hash_files=not args.no_file_hashes)
    if args.output:
        _write_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "ready_with_declared_gaps" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
