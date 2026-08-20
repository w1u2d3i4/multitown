"""Build and audit the canonical sanitized public source archive.

The internal evidence checkout contains frozen records that are intentionally
not part of a public source release.  This module classifies every tracked path
as public or excluded, archives only the public paths from an exact clean Git
commit, and then audits the resulting tar without extracting it.

This is a source-surface and packaging control.  It does not sanitize Git
history, authenticate a builder, sign an artifact, or reproduce an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import io
import json
import os
import posixpath
import re
import stat
import subprocess
import tarfile
import unicodedata
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any

PUBLIC_RELEASE_AUDIT_VERSION = "multitown-public-release-audit-v1"
PUBLIC_ARCHIVE_PROFILE = "multitown-canonical-git-tar-v1"
ARCHIVE_PREFIX = "multitown-bench"
PUBLIC_INCLUDE_MANIFEST = "release/public-include.txt"
PUBLIC_EXCLUDE_MANIFEST = "release/public-exclude.txt"
PUBLIC_EXECUTABLE_MANIFEST = "release/public-executable.txt"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 10_000

_FORBIDDEN_TOP_LEVEL = frozenset(
    {".git", "artifacts", "interrupted-attempts", "third_party"}
)
_FORBIDDEN_SUFFIXES = frozenset({".ckpt", ".pickle", ".pkl", ".pt", ".pth"})
_ALLOWED_PUBLIC_RECORDS = frozenset(
    {
        "records/a22-adaptive-formal-20260814/RESULTS.md",
        "records/a22-adaptive-formal-20260814/artifact-manifest.json",
        "records/a22-adaptive-formal-20260814/outer-comparison.png",
        "records/a22-adaptive-formal-20260814/record.json",
        "records/a22-adaptive-formal-20260814/report-summary.json",
        "records/a22-adaptive-formal-20260814/training-diagnostics.png",
        "records/a27-e-same-bank-tail-20260815/RESULTS.md",
        "records/a27-e-same-bank-tail-20260815/artifact-manifest.json",
        "records/a27-e-same-bank-tail-20260815/record.json",
        "records/a9-offline-fitted-q-20260812/RESULTS.md",
        "records/a9-v2-ppo-oof-20260813-r2/RESULTS.md",
        "records/a9-v2-ppo-oof-20260813-r2/artifact-manifest.json",
        "records/a9-v2-ppo-oof-20260813-r2/record.json",
        "records/a9-v3-review-shield-diagnostic-20260813/RESULTS.md",
        "records/a9-v3-review-shield-diagnostic-20260813/artifact-manifest.json",
        "records/a9-v3-review-shield-diagnostic-20260813/record.json",
        "records/formal-a78-20260810T183300Z/RESULTS.md",
        "records/formal-a78-20260810T183300Z/a7-online-summary.json",
        "records/formal-a78-20260810T183300Z/a8-online-summary.json",
        "records/formal-a78-20260810T183300Z/overall.csv",
        "records/reproductions/20260810/masbench/RESULTS.md",
        "records/reproductions/20260810/masbench/axis-accuracy.png",
        "records/reproductions/20260810/masbench/comparison.json",
        "records/reproductions/20260810/masbench/cumulative-tokens.png",
        "records/reproductions/20260810/masbench/latency-ecdf.png",
        "records/reproductions/20260810/masbench/routing-summary.json",
        "records/reproductions/20260810/masbench/tradeoff.png",
        "records/reproductions/20260810/silo-subset/RESULTS.md",
        "records/reproductions/20260810/silo-subset/outcomes.png",
        "records/reproductions/20260810/silo-subset/summary.json",
        "records/reproductions/20260810/silo-subset/system-curves.png",
        "records/reproductions/20260810/silo-subset/tokens.png",
        "records/serving-traces/a8-formal-v0.2.0-export-manifest.json",
        "records/third-party-audit.json",
        "records/third-party-lock.json",
    }
)
_COMPACT_RECORD_MANIFESTS = frozenset(
    {
        "records/a22-adaptive-formal-20260814/artifact-manifest.json",
        "records/a27-e-same-bank-tail-20260815/artifact-manifest.json",
        "records/a9-v2-ppo-oof-20260813-r2/artifact-manifest.json",
        "records/a9-v3-review-shield-diagnostic-20260813/artifact-manifest.json",
    }
)
_REQUIRED_FILES = (
    frozenset(
        {
            ".gitattributes",
            ".github/workflows/ci.yml",
            "DATA_CARD.md",
            "DATA_LICENSE",
            "LICENSE",
            "PYPI_README.md",
            "README.md",
            "benchmarks/external/masbench-v1/LICENSE",
            "benchmarks/external/masbench-v1/README.md",
            "docs/A24_PORTABLE_FIXTURE.md",
            "docs/A25_PORTABLE_PROFILE_V1.md",
            "docs/PUBLIC_RELEASE.md",
            "examples/a25-portable-sample-v1/README.md",
            "examples/a25-portable-sample-v1/bundle/common-state.safetensors",
            "examples/a25-portable-sample-v1/bundle/contract.json",
            "examples/a25-portable-sample-v1/bundle/manifest.json",
            "examples/a25-portable-sample-v1/bundle/observations.safetensors",
            "multitown/a25_portable_checker.py",
            "multitown/fixtures/a24_portable_policy_v1.json",
            "multitown/fixtures/a24_portable_receipt_v1.schema.json",
            "multitown/fixtures/a24_portable_v1.json",
            "multitown/public_release.py",
            "pyproject.toml",
            PUBLIC_EXCLUDE_MANIFEST,
            PUBLIC_EXECUTABLE_MANIFEST,
            PUBLIC_INCLUDE_MANIFEST,
            "schemas/a24-semantic-verifier-receipt-v1.schema.json",
            "schemas/a25-portable-contract-v1.schema.json",
            "schemas/a25-portable-manifest-v1.schema.json",
            "schemas/a25-portable-report-v1.schema.json",
            "tests/test_a25_portable_checker.py",
            "tests/test_public_release.py",
            "tools/generate_a25_portable_sample.py",
            "verification/a24-semantic-v1-policy.json",
        }
    )
    | _ALLOWED_PUBLIC_RECORDS
)
_REQUIRED_SHA256 = {
    "benchmarks/external/masbench-v1/LICENSE": (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
}
_CI_NEGATIVE_SCANNER_FRAGMENT = (
    b'rb"/home/' + b'dilab|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|"'
)
_HOST_PATH_PATTERNS = (
    re.compile(rb"/(?:home|Users)/[A-Za-z0-9._-]+(?:/|\b)"),
    re.compile(rb"/" + rb"root(?:/|\b)"),
    re.compile(rb"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._-]+(?:[\\/]|\b)"),
)
_PRIVATE_ENDPOINT_PATTERN = re.compile(
    rb"(?i)(?:\blocalhost\b|"
    rb"(?:127|10)(?:\.[0-9]{1,3}){3}|"
    rb"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}|"
    rb"192\.168(?:\.[0-9]{1,3}){2}|"
    rb"169\.254(?:\.[0-9]{1,3}){2}|"
    rb"0\.0\.0\.0|\[?::1\]?|"
    rb"\[?f[cd][0-9a-f:]*:[0-9a-f:]+\]?|"
    rb"\[?fe[89ab][0-9a-f:]*:[0-9a-f:]+\]?)"
)
_SECRET_PATTERNS = (
    re.compile(rb"-{5}BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-{5}"),
    re.compile(rb"-{5}BEGIN PGP PRIVATE KEY BLOCK-{5}"),
    re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    re.compile(rb"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{24,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
)
_MARKDOWN_INLINE_LINK_PATTERN = re.compile(r"!?\[[^\]\r\n]*\]\(([^)\r\n]+)\)")
_MARKDOWN_REFERENCE_LINK_PATTERN = re.compile(
    r"(?m)^ {0,3}\[([^\]\r\n]+)\]:[ \t]*([^\s]+)"
)
_MARKDOWN_REFERENCE_USE_PATTERN = re.compile(r"!?\[([^\]\r\n]+)\]\[([^\]\r\n]*)\]")
_MARKDOWN_AUTOLINK_SCHEME_PATTERN = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]*:[^<>\s]*)>")
_MARKDOWN_MULTILINE_LINK_PATTERN = re.compile(r"\]\([ \t]*\r?\n")
_ALLOWED_EXTERNAL_LINK_SCHEMES = frozenset({"http", "https", "mailto"})
_HTML_LINK_ATTRIBUTES = frozenset(
    {
        "action",
        "cite",
        "data",
        "formaction",
        "href",
        "longdesc",
        "manifest",
        "poster",
        "profile",
        "src",
        "usemap",
        "xlink:href",
    }
)
_HTML_LINK_LIST_ATTRIBUTES = frozenset({"ping"})
_HTML_SRCSET_ATTRIBUTES = frozenset({"imagesrcset", "srcset"})


class PublicReleaseAuditError(RuntimeError):
    """The source tree or generated public archive failed closed."""


def _reject(condition: bool, message: str) -> None:
    if not condition:
        raise PublicReleaseAuditError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_path(raw: str, *, label: str) -> str:
    _reject(type(raw) is str and bool(raw), f"empty path in {label}")
    _reject("\\" not in raw and "\x00" not in raw, f"unsafe path in {label}: {raw!r}")
    _reject(
        all(ord(character) >= 32 and ord(character) != 127 for character in raw),
        f"control character in {label}: {raw!r}",
    )
    path = PurePosixPath(raw)
    _reject(
        unicodedata.normalize("NFC", raw) == raw, f"non-NFC path in {label}: {raw!r}"
    )
    _reject(not path.is_absolute(), f"absolute path in {label}: {raw!r}")
    _reject(
        raw == path.as_posix()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"non-canonical path in {label}: {raw!r}",
    )
    return path.as_posix()


def _parse_path_manifest(payload: bytes, *, label: str) -> tuple[str, ...]:
    _reject(0 < len(payload) <= 4 * 1024 * 1024, f"unbounded {label}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicReleaseAuditError(f"invalid UTF-8 in {label}") from exc
    _reject(text.endswith("\n"), f"{label} must end with a newline")
    raw_lines = text.splitlines()
    _reject(raw_lines and all(raw_lines), f"blank line in {label}")
    paths = tuple(_canonical_path(line, label=label) for line in raw_lines)
    _reject(paths == tuple(sorted(paths)), f"{label} is not bytewise sorted")
    _reject(len(paths) == len(set(paths)), f"duplicate path in {label}")
    return paths


def _parse_export_ignores(payload: bytes) -> tuple[str, ...]:
    _reject(0 < len(payload) <= 1024 * 1024, "unbounded .gitattributes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicReleaseAuditError("invalid UTF-8 in .gitattributes") from exc
    paths: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        _reject(
            len(fields) == 2 and fields[1] == "export-ignore",
            f"unsupported .gitattributes rule on line {line_number}",
        )
        paths.append(_canonical_path(fields[0], label=".gitattributes"))
    result = tuple(paths)
    _reject(result == tuple(sorted(result)), ".gitattributes rules are not sorted")
    _reject(len(result) == len(set(result)), "duplicate .gitattributes rule")
    return result


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(root: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "tar.umask=0022",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError as exc:
        raise PublicReleaseAuditError("cannot execute Git") from exc
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PublicReleaseAuditError(
            f"Git command failed ({arguments[0]}): {error or completed.returncode}"
        )
    return completed.stdout


def _tracked_paths(root: Path, revision: str) -> tuple[dict[str, str], int]:
    payload = _run_git(root, "ls-tree", "-r", "-z", "--full-tree", revision)
    records: dict[str, str] = {}
    for raw_record in payload.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", maxsplit=1)
            mode, object_type, _object_id = metadata.decode("ascii").split()
            path = _canonical_path(raw_path.decode("utf-8"), label="Git tree")
        except (UnicodeDecodeError, ValueError) as exc:
            raise PublicReleaseAuditError("malformed Git tree record") from exc
        _reject(
            object_type == "blob" and mode in {"100644", "100755"},
            f"non-regular tracked path: {path}",
        )
        _reject(path not in records, f"duplicate tracked path: {path}")
        records[path] = mode
    _reject(bool(records), "Git tree is empty")
    return records, len(payload)


def _relative_archive_path(member_name: str) -> str:
    normalized = member_name.removesuffix("/")
    path = PurePosixPath(normalized)
    _reject(
        path.parts and path.parts[0] == ARCHIVE_PREFIX,
        f"archive member is outside the fixed prefix: {member_name!r}",
    )
    _reject(
        len(path.parts) >= 2, f"archive member has no relative path: {member_name!r}"
    )
    relative = _canonical_path(
        PurePosixPath(*path.parts[1:]).as_posix(), label="archive"
    )
    _reject(
        normalized == f"{ARCHIVE_PREFIX}/{relative}",
        f"non-canonical archive member name: {member_name!r}",
    )
    return relative


def _scan_payload(path: str, payload: bytes) -> None:
    scan_payload = payload
    if _CI_NEGATIVE_SCANNER_FRAGMENT in scan_payload:
        _reject(
            path == ".github/workflows/ci.yml"
            and scan_payload.count(_CI_NEGATIVE_SCANNER_FRAGMENT) == 1,
            f"unapproved scanner exception in public archive: {path}",
        )
        scan_payload = scan_payload.replace(_CI_NEGATIVE_SCANNER_FRAGMENT, b"")
    for pattern in _HOST_PATH_PATTERNS:
        _reject(
            pattern.search(scan_payload) is None,
            f"host-local path leaked into public archive: {path}",
        )
    for pattern in _SECRET_PATTERNS:
        _reject(
            pattern.search(scan_payload) is None, f"credential-like content: {path}"
        )
    if path.startswith("records/"):
        _reject(
            _PRIVATE_ENDPOINT_PATTERN.search(scan_payload) is None,
            f"private endpoint in public record: {path}",
        )


def _markdown_link_path(source_path: str, raw_target: str) -> tuple[str | None, bool]:
    target = raw_target.strip()
    _reject(bool(target), f"empty Markdown link target: {source_path}")
    if target.startswith("<"):
        closing = target.find(">")
        _reject(closing > 1, f"malformed Markdown link target: {source_path}")
        target = target[1:closing]
    else:
        target = target.split(maxsplit=1)[0]
    _reject(
        all(ord(character) >= 32 and ord(character) != 127 for character in target),
        f"control character in Markdown link: {source_path}",
    )
    if target.startswith("#"):
        return None, False
    scheme = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", target)
    if scheme is not None:
        _reject(
            scheme.group(1).lower() in _ALLOWED_EXTERNAL_LINK_SCHEMES,
            f"unsupported Markdown link scheme in {source_path}: {target!r}",
        )
        return None, True
    if target.startswith("//"):
        return None, True

    path_target = urllib.parse.unquote(target.split("#", maxsplit=1)[0])
    path_target = path_target.split("?", maxsplit=1)[0]
    _reject(bool(path_target), f"empty local Markdown link: {source_path}")
    _reject(
        not path_target.startswith("/") and "\\" not in path_target,
        f"unsafe local Markdown link in {source_path}: {target!r}",
    )
    source_parent = PurePosixPath(source_path).parent.as_posix()
    normalized = posixpath.normpath(posixpath.join(source_parent, path_target))
    _reject(
        normalized not in {".", ".."} and not normalized.startswith("../"),
        f"Markdown link escapes the public archive in {source_path}: {target!r}",
    )
    return _canonical_path(normalized, label=f"Markdown link in {source_path}"), False


def _reference_label(raw_label: str) -> str:
    return " ".join(raw_label.split()).casefold()


class _MarkdownHTMLLinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []
        self.unsupported_attributes: list[str] = []

    def handle_starttag(
        self, _tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attributes:
            normalized_name = name.casefold()
            if normalized_name in _HTML_LINK_ATTRIBUTES and value is not None:
                self.targets.append(value)
            elif normalized_name in _HTML_LINK_LIST_ATTRIBUTES and value is not None:
                self.targets.extend(value.split())
            elif normalized_name in _HTML_SRCSET_ATTRIBUTES and value is not None:
                self.targets.extend(
                    candidate.strip().split(maxsplit=1)[0]
                    for candidate in value.split(",")
                    if candidate.strip()
                )
            elif value is not None:
                self.unsupported_attributes.append(normalized_name)

    def handle_startendtag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attributes)


def _audit_markdown_links(files: dict[str, bytes]) -> dict[str, int]:
    local_count = 0
    external_count = 0
    for source_path in sorted(path for path in files if path.endswith(".md")):
        try:
            text = files[source_path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PublicReleaseAuditError(
                f"invalid UTF-8 Markdown file: {source_path}"
            ) from exc
        _reject(
            _MARKDOWN_MULTILINE_LINK_PATTERN.search(text) is None,
            f"unsupported multiline Markdown link in {source_path}",
        )
        definitions: dict[str, str] = {}
        for match in _MARKDOWN_REFERENCE_LINK_PATTERN.finditer(text):
            label = _reference_label(match.group(1))
            _reject(
                bool(label) and label not in definitions,
                f"duplicate/empty Markdown reference label: {source_path}",
            )
            definitions[label] = match.group(2)
        for match in _MARKDOWN_REFERENCE_USE_PATTERN.finditer(text):
            label = _reference_label(match.group(2) or match.group(1))
            _reject(
                label in definitions,
                f"undefined Markdown reference link in {source_path}: {label!r}",
            )

        html_links = _MarkdownHTMLLinkParser()
        try:
            html_links.feed(text)
            html_links.close()
        except (AssertionError, ValueError) as exc:
            raise PublicReleaseAuditError(
                f"malformed raw HTML in Markdown file: {source_path}"
            ) from exc
        _reject(
            not html_links.unsupported_attributes,
            f"unsupported Markdown raw HTML attribute in {source_path}: "
            f"{html_links.unsupported_attributes[:1]!r}",
        )

        raw_targets = [
            match.group(1) for match in _MARKDOWN_INLINE_LINK_PATTERN.finditer(text)
        ]
        for raw_target in raw_targets:
            _reject(
                "(" not in raw_target,
                f"unsupported nested Markdown link destination in {source_path}",
            )
        raw_targets.extend(definitions.values())
        raw_targets.extend(html_links.targets)
        raw_targets.extend(
            match.group(1) for match in _MARKDOWN_AUTOLINK_SCHEME_PATTERN.finditer(text)
        )
        for raw_target in raw_targets:
            target, external = _markdown_link_path(source_path, raw_target)
            if external:
                external_count += 1
                continue
            if target is None:
                continue
            local_count += 1
            target_prefix = f"{target.rstrip('/')}/"
            _reject(
                target in files
                or any(candidate.startswith(target_prefix) for candidate in files),
                f"missing public Markdown link target in {source_path}: {raw_target!r}",
            )
    return {
        "external_link_count": external_count,
        "local_link_count": local_count,
    }


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _reject(
                type(key) is str and key not in result, f"duplicate JSON key: {label}"
            )
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicReleaseAuditError(f"invalid JSON: {label}") from exc
    _reject(type(value) is dict, f"JSON value is not an object: {label}")
    return value


def _audit_compact_record_manifest(
    manifest_path: str, manifest_payload: bytes, files: dict[str, bytes]
) -> None:
    manifest = _strict_json_object(manifest_payload, label=manifest_path)
    _reject(
        manifest.get("schema_version") == "multitown-artifact-manifest-v1",
        f"unexpected compact record manifest schema: {manifest_path}",
    )
    _reject(
        manifest.get("source_dirty") is False,
        f"dirty compact record manifest: {manifest_path}",
    )
    entries = manifest.get("files")
    roots = manifest.get("artifact_roots")
    _reject(
        type(entries) is list and bool(entries), f"invalid files list: {manifest_path}"
    )
    _reject(type(roots) is list and len(roots) == 1, f"invalid roots: {manifest_path}")
    root = roots[0]
    _reject(type(root) is dict, f"invalid root object: {manifest_path}")
    expected_root = PurePosixPath(manifest_path).parent.as_posix()
    _reject(root.get("path") == expected_root, f"wrong compact root: {manifest_path}")

    seen: set[str] = set()
    total_bytes = 0
    for entry in entries:
        _reject(type(entry) is dict, f"invalid compact entry: {manifest_path}")
        _reject(
            set(entry) == {"bytes", "path", "sha256"},
            f"unexpected compact entry fields: {manifest_path}",
        )
        path = _canonical_path(entry["path"], label=manifest_path)
        _reject(path.startswith(f"{expected_root}/"), f"escaped compact path: {path}")
        _reject(
            path not in seen and path in files,
            f"missing/duplicate compact path: {path}",
        )
        seen.add(path)
        payload = files[path]
        _reject(
            type(entry["bytes"]) is int
            and entry["bytes"] == len(payload)
            and type(entry["sha256"]) is str
            and entry["sha256"] == _sha256(payload),
            f"compact record digest mismatch: {path}",
        )
        total_bytes += len(payload)
    _reject(
        type(root.get("file_count")) is int
        and root["file_count"] == len(entries)
        and type(root.get("byte_count")) is int
        and root["byte_count"] == total_bytes,
        f"compact record aggregate mismatch: {manifest_path}",
    )
    expected_paths = {
        path
        for path in files
        if path.startswith(f"{expected_root}/") and path != manifest_path
    }
    _reject(
        seen == expected_paths,
        f"compact record manifest is not a closed inventory: {manifest_path}",
    )


def _pax_comment_record(source_commit: str) -> bytes:
    payload = f"comment={source_commit}\n".encode("ascii")
    length = len(payload) + 2
    while True:
        record = f"{length} ".encode("ascii") + payload
        if len(record) == length:
            return record
        length = len(record)


def _tar_text_field(header: bytes, start: int, size: int, *, label: str) -> str:
    field = header[start : start + size]
    value, separator, padding = field.partition(b"\0")
    _reject(separator == b"\0" and not padding.strip(b"\0"), f"unsafe {label}")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicReleaseAuditError(f"invalid UTF-8 in {label}") from exc


def _tar_octal_field(header: bytes, start: int, size: int, *, label: str) -> int:
    field = header[start : start + size]
    _reject(
        len(field) == size
        and field[-1:] == b"\0"
        and all(character in b"01234567" for character in field[:-1]),
        f"non-canonical octal {label}",
    )
    return int(field[:-1], 8)


def _reject_noncanonical_checksum(header: bytes, *, label: str) -> None:
    field = header[148:156]
    _reject(
        re.fullmatch(rb"[0-7]{7}\0", field) is not None,
        f"non-canonical tar checksum field: {label}",
    )


def _audit_git_global_pax_header(archive_bytes: bytes, source_commit: str) -> int:
    _reject(len(archive_bytes) >= 1024, "truncated Git global pax header")
    header = archive_bytes[:512]
    record = _pax_comment_record(source_commit)
    _reject(
        _tar_text_field(header, 0, 100, label="global pax name") == "pax_global_header",
        "unexpected Git global pax header name",
    )
    _reject(header[156:157] == tarfile.XGLTYPE, "missing Git global pax type")
    _reject(header[257:265] == b"ustar\x0000", "invalid Git global pax magic")
    _reject_noncanonical_checksum(header, label="global pax")
    _reject(
        _tar_octal_field(header, 100, 8, label="global pax mode") == 0o666
        and _tar_octal_field(header, 108, 8, label="global pax uid") == 0
        and _tar_octal_field(header, 116, 8, label="global pax gid") == 0
        and _tar_octal_field(header, 124, 12, label="global pax size") == len(record)
        and _tar_text_field(header, 265, 32, label="global pax uname") == "root"
        and _tar_text_field(header, 297, 32, label="global pax gname") == "root"
        and _tar_octal_field(header, 329, 8, label="global pax devmajor") == 0
        and _tar_octal_field(header, 337, 8, label="global pax devminor") == 0
        and _tar_text_field(header, 345, 155, label="global pax prefix") == ""
        and header[500:512] == b"\0" * 12,
        "unexpected Git global pax metadata",
    )
    _scan_payload("<tar-global-pax-header>", header)
    _reject(
        archive_bytes[512 : 512 + len(record)] == record
        and archive_bytes[512 + len(record) : 1024] == b"\0" * (512 - len(record)),
        "non-canonical Git global pax payload or padding",
    )
    return _tar_octal_field(header, 136, 12, label="global pax mtime")


def _audit_raw_member_header(
    archive_bytes: bytes, member: tarfile.TarInfo, source_commit: str
) -> int:
    _reject(
        member.offset_data == member.offset + 512,
        f"unexpected extended header before archive member: {member.name!r}",
    )
    header = archive_bytes[member.offset : member.offset_data]
    _reject(len(header) == 512, f"truncated tar header: {member.name!r}")
    expected_name = f"{member.name}/" if member.isdir() else member.name
    expected_type = tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
    _reject_noncanonical_checksum(header, label=member.name)
    _reject(
        _tar_text_field(header, 0, 100, label="member name") == expected_name
        and _tar_octal_field(header, 100, 8, label="member mode") == member.mode
        and _tar_octal_field(header, 108, 8, label="member uid") == member.uid == 0
        and _tar_octal_field(header, 116, 8, label="member gid") == member.gid == 0
        and _tar_octal_field(header, 124, 12, label="member size") == member.size
        and _tar_octal_field(header, 136, 12, label="member mtime") == member.mtime
        and header[156:157] == expected_type
        and header[257:265] == b"ustar\x0000"
        and _tar_text_field(header, 157, 100, label="member linkname") == ""
        and _tar_text_field(header, 265, 32, label="member uname") == "root"
        and _tar_text_field(header, 297, 32, label="member gname") == "root"
        and _tar_text_field(header, 345, 155, label="member prefix") == ""
        and _tar_octal_field(header, 329, 8, label="member devmajor")
        == member.devmajor
        == 0
        and _tar_octal_field(header, 337, 8, label="member devminor")
        == member.devminor
        == 0
        and header[500:512] == b"\0" * 12,
        f"non-canonical raw tar header: {member.name!r}",
    )
    _scan_payload(f"<tar-member-header:{member.name}>", header)
    padded_end = ((member.offset_data + member.size + 511) // 512) * 512
    _reject(
        archive_bytes[member.offset_data + member.size : padded_end]
        == b"\0" * (padded_end - member.offset_data - member.size),
        f"non-zero tar payload padding: {member.name!r}",
    )
    _reject(
        member.pax_headers == {"comment": source_commit},
        f"unexpected member pax metadata: {member.name!r}",
    )
    return padded_end


def _audit_archive_bytes(archive_bytes: bytes) -> dict[str, Any]:
    _reject(
        0 < len(archive_bytes) <= MAX_ARCHIVE_BYTES,
        "public source archive is empty or exceeds the byte limit",
    )
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    directories: set[str] = set()
    root_directory_seen = False
    source_commit = ""
    source_mtime: int | None = None
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            _reject(
                set(archive.pax_headers) == {"comment"}
                and re.fullmatch(
                    r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                    archive.pax_headers["comment"],
                )
                is not None,
                "missing or malformed Git commit identity in tar pax headers",
            )
            source_commit = archive.pax_headers["comment"]
            global_mtime = _audit_git_global_pax_header(archive_bytes, source_commit)
            member_count = 0
            expected_member_offset = 1024
            for member in archive:
                member_count += 1
                _reject(member_count <= MAX_FILES * 2, "too many archive members")
                _reject(
                    member.offset == expected_member_offset,
                    f"unexpected tar header gap or extension: {member.name!r}",
                )
                _reject(
                    member.isdir() or member.isreg(),
                    f"non-regular archive member: {member.name}",
                )
                expected_member_offset = _audit_raw_member_header(
                    archive_bytes, member, source_commit
                )
                _reject(
                    member.uid == 0
                    and member.gid == 0
                    and member.uname == "root"
                    and member.gname == "root"
                    and member.linkname == ""
                    and member.devmajor == 0
                    and member.devminor == 0,
                    f"unexpected archive member identity metadata: {member.name!r}",
                )
                _reject(
                    type(member.mtime) is int and member.mtime >= 0,
                    f"unexpected archive member timestamp: {member.name!r}",
                )
                if source_mtime is None:
                    source_mtime = member.mtime
                _reject(
                    member.mtime == source_mtime,
                    f"inconsistent archive member timestamp: {member.name!r}",
                )
                if member.isdir():
                    _reject(
                        member.size == 0, f"non-empty archive directory: {member.name}"
                    )
                    normalized = member.name.removesuffix("/")
                    if normalized == ARCHIVE_PREFIX:
                        _reject(
                            not root_directory_seen,
                            "duplicate archive root directory",
                        )
                        _reject(
                            member.mode == 0o755,
                            "unexpected archive root directory mode",
                        )
                        root_directory_seen = True
                        continue
                    directory_path = _relative_archive_path(member.name)
                    _reject(
                        directory_path not in directories,
                        f"duplicate archive directory: {directory_path}",
                    )
                    _reject(
                        member.mode == 0o755,
                        f"unexpected directory mode: {member.name}",
                    )
                    directories.add(directory_path)
                    continue
                path = _relative_archive_path(member.name)
                _reject(path not in files, f"duplicate archive member: {path}")
                _reject(
                    0 <= member.size <= MAX_FILE_BYTES,
                    f"archive member exceeds the byte limit: {path}",
                )
                _reject(member.mode in {0o644, 0o755}, f"unexpected file mode: {path}")
                stream = archive.extractfile(member)
                _reject(stream is not None, f"cannot read archive member: {path}")
                payload = stream.read(MAX_FILE_BYTES + 1)
                _reject(len(payload) == member.size, f"archive size mismatch: {path}")
                files[path] = payload
                modes[path] = member.mode
            _reject(
                source_mtime is not None and global_mtime == source_mtime,
                "global pax timestamp differs from archive member timestamps",
            )
            _reject(
                archive.offset == expected_member_offset,
                "tar parser logical end differs from audited member layout",
            )
            minimum_size = expected_member_offset + 1024
            expected_archive_size = (
                (minimum_size + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
            ) * tarfile.RECORDSIZE
            _reject(
                len(archive_bytes) == expected_archive_size
                and archive_bytes[expected_member_offset:]
                == b"\0" * (expected_archive_size - expected_member_offset),
                "non-canonical or non-zero tar end-of-archive trailer",
            )
    except (tarfile.TarError, OSError) as exc:
        raise PublicReleaseAuditError("invalid public source tar archive") from exc

    _reject(0 < len(files) <= MAX_FILES, "invalid public archive file count")
    _reject(root_directory_seen, "public archive root directory is missing")
    missing = sorted(_REQUIRED_FILES - files.keys())
    _reject(not missing, f"required public files are missing: {missing}")

    include_paths = _parse_path_manifest(
        files[PUBLIC_INCLUDE_MANIFEST], label=PUBLIC_INCLUDE_MANIFEST
    )
    exclude_paths = _parse_path_manifest(
        files[PUBLIC_EXCLUDE_MANIFEST], label=PUBLIC_EXCLUDE_MANIFEST
    )
    executable_paths = _parse_path_manifest(
        files[PUBLIC_EXECUTABLE_MANIFEST], label=PUBLIC_EXECUTABLE_MANIFEST
    )
    _reject(
        set(include_paths).isdisjoint(exclude_paths),
        "public include/exclude manifests overlap",
    )
    _reject(
        tuple(sorted(files)) == include_paths,
        "archive inventory does not exactly match the public include manifest",
    )
    expected_directories: set[str] = set()
    for path in include_paths:
        parts = PurePosixPath(path).parts
        expected_directories.update(
            PurePosixPath(*parts[:end]).as_posix() for end in range(1, len(parts))
        )
    _reject(
        set(include_paths).isdisjoint(expected_directories),
        "public path is both a file and a directory",
    )
    _reject(
        directories == expected_directories,
        "archive directory inventory does not match public file parents",
    )
    _reject(
        set(executable_paths) <= set(include_paths),
        "public executable manifest contains a non-public path",
    )
    export_ignores = _parse_export_ignores(files[".gitattributes"])
    _reject(
        export_ignores == exclude_paths,
        ".gitattributes export-ignore rules do not match the exclusion manifest",
    )
    _reject(
        set(exclude_paths).isdisjoint(files),
        "an excluded path is present in the public archive",
    )
    public_records = {path for path in files if path.startswith("records/")}
    _reject(
        public_records == _ALLOWED_PUBLIC_RECORDS,
        "public record inventory differs from the reviewed exact allowlist",
    )
    markdown_links = _audit_markdown_links(files)
    for manifest_path in _COMPACT_RECORD_MANIFESTS:
        _audit_compact_record_manifest(manifest_path, files[manifest_path], files)

    casefold_paths: dict[str, str] = {}
    for path in sorted(expected_directories):
        collision_key = unicodedata.normalize("NFC", path).casefold()
        previous_path = casefold_paths.get(collision_key)
        _reject(
            previous_path is None or previous_path == path,
            f"case/Unicode-colliding public paths: {previous_path!r}, {path!r}",
        )
        casefold_paths[collision_key] = path
    inventory_rows: list[dict[str, Any]] = []
    for path in include_paths:
        path_object = PurePosixPath(path)
        collision_key = unicodedata.normalize("NFC", path).casefold()
        previous_path = casefold_paths.get(collision_key)
        _reject(
            previous_path is None or previous_path == path,
            f"case/Unicode-colliding public paths: {previous_path!r}, {path!r}",
        )
        casefold_paths[collision_key] = path
        _reject(
            set(path_object.parts).isdisjoint(_FORBIDDEN_TOP_LEVEL | {"interrupted"}),
            f"forbidden public path segment: {path}",
        )
        _reject(
            path_object.suffix.lower() not in _FORBIDDEN_SUFFIXES,
            f"checkpoint-like file in public archive: {path}",
        )
        _reject(
            not path_object.name.startswith(".env"),
            f"environment file in public archive: {path}",
        )
        payload = files[path]
        expected_mode = 0o755 if path in executable_paths else 0o644
        _reject(modes[path] == expected_mode, f"unexpected executable mode: {path}")
        _scan_payload(path, payload)
        digest = _sha256(payload)
        expected_digest = _REQUIRED_SHA256.get(path)
        _reject(
            expected_digest is None or digest == expected_digest,
            f"required public file digest mismatch: {path}",
        )
        inventory_rows.append(
            {
                "bytes": len(payload),
                "mode": f"{modes[path]:04o}",
                "path": path,
                "sha256": digest,
            }
        )

    inventory_bytes = json.dumps(
        inventory_rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "archive_bytes": len(archive_bytes),
        "archive_profile": PUBLIC_ARCHIVE_PROFILE,
        "archive_sha256": _sha256(archive_bytes),
        "directory_count": len(directories) + 1,
        "excluded_count": len(exclude_paths),
        "excluded_paths_sha256": _sha256(files[PUBLIC_EXCLUDE_MANIFEST]),
        "file_count": len(files),
        "included_paths_sha256": _sha256(files[PUBLIC_INCLUDE_MANIFEST]),
        "inventory_sha256": _sha256(inventory_bytes),
        "inventory": inventory_rows,
        "markdown_links": markdown_links,
        "embedded_commit_claim": source_commit,
        "source_mtime": source_mtime,
        "public_paths": include_paths,
        "excluded_paths": exclude_paths,
    }


_OutputIdentity = tuple[str, int, int, str]


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _unlink_owned_entry(parent_fd: int, identity: _OutputIdentity) -> str | None:
    name, device, inode, label = identity
    try:
        current = _entry_stat(parent_fd, name)
    except OSError as exc:
        return f"failed to inspect {label} {name} before cleanup: {exc}"
    if current is None:
        return None
    if (current.st_dev, current.st_ino) != (device, inode):
        return f"{label} identity changed before cleanup: {name}"
    try:
        os.unlink(name, dir_fd=parent_fd)
    except OSError as exc:
        return f"failed to clean {label} {name}: {exc}"
    return None


def _write_staged_output(
    parent_fd: int, payload: bytes, *, label: str, ordinal: int
) -> _OutputIdentity:
    descriptor: int | None = None
    identity: _OutputIdentity | None = None
    created_staging_name: str | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for attempt in range(128):
            staging_name = (
                f".multitown-release-{os.getpid()}-{ordinal}-{attempt}-"
                f"{os.urandom(12).hex()}.tmp"
            )
            try:
                descriptor = os.open(
                    staging_name,
                    flags,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            created_staging_name = staging_name
            opened = os.fstat(descriptor)
            identity = (staging_name, opened.st_dev, opened.st_ino, label)
            break
        _reject(identity is not None, f"could not allocate private {label} staging")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _reject(written > 0, f"zero-byte {label} write")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        _reject(
            stat.S_ISREG(opened.st_mode)
            and stat.S_IMODE(opened.st_mode) == 0o600
            and opened.st_nlink == 1
            and opened.st_size == len(payload),
            f"unexpected staged {label} identity",
        )
        os.close(descriptor)
        descriptor = None
        return identity
    except BaseException as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        cleanup_error = (
            _unlink_owned_entry(parent_fd, identity) if identity is not None else None
        )
        if identity is None and created_staging_name is not None:
            try:
                os.unlink(created_staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                cleanup_error = (
                    f"failed to clean unverified {label} staging "
                    f"{created_staging_name}: {cleanup_exc}"
                )
        if cleanup_error is not None:
            raise PublicReleaseAuditError(
                f"public release cleanup incomplete: {cleanup_error}"
            ) from exc
        raise


def _publish_staged_output(
    parent_fd: int,
    staged: _OutputIdentity,
    final_name: str,
    *,
    published: list[_OutputIdentity],
) -> None:
    staging_name, device, inode, label = staged
    final_identity: _OutputIdentity | None = None
    try:
        os.link(
            staging_name,
            final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        final_identity = (final_name, device, inode, label)
        published.append(final_identity)
        final_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        _reject(
            (final_stat.st_dev, final_stat.st_ino) == (device, inode),
            f"published {label} identity mismatch",
        )
        cleanup_error = _unlink_owned_entry(parent_fd, staged)
        _reject(cleanup_error is None, cleanup_error or f"failed to publish {label}")
    except BaseException as exc:
        cleanup_error = (
            _unlink_owned_entry(parent_fd, final_identity)
            if final_identity is not None
            else None
        )
        if cleanup_error is not None:
            raise PublicReleaseAuditError(
                f"public release cleanup incomplete: {cleanup_error}"
            ) from exc
        raise


def _write_private_outputs_once(
    outputs: list[tuple[Path, bytes, str]], *, root: Path
) -> None:
    _reject(bool(outputs), "no public release outputs requested")
    resolved: list[tuple[Path, str, bytes, str]] = []
    for output, payload, label in outputs:
        parent = output.parent.resolve(strict=True)
        _reject(parent.is_dir(), f"{label} output parent is not a directory")
        _reject(output.name not in {"", ".", ".."}, f"invalid {label} output name")
        resolved_output = parent / output.name
        _reject(
            root != resolved_output and root not in resolved_output.parents,
            f"{label} output must be outside the source repository",
        )
        resolved.append((parent, output.name, payload, label))
    parents = {parent for parent, _, _, _ in resolved}
    _reject(
        len(parents) == 1,
        "public archive and audit report must share one output directory",
    )
    names = [name for _, name, _, _ in resolved]
    _reject(len(names) == len(set(names)), "public release outputs must be distinct")

    parent = resolved[0][0]
    before = parent.stat(follow_symlinks=False)
    parent_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd: int | None = None
    staged: list[_OutputIdentity] = []
    published: list[_OutputIdentity] = []
    try:
        parent_fd = os.open(parent, parent_flags)
        opened = os.fstat(parent_fd)
        _reject(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            "public release output directory changed before open",
        )
        _reject(
            opened.st_uid == os.geteuid() and stat.S_IMODE(opened.st_mode) & 0o022 == 0,
            "public release output directory must be owned by the current euid "
            "and not group/other writable",
        )
        _reject(
            all(_entry_stat(parent_fd, name) is None for name in names),
            "public release output already exists",
        )
        for ordinal, (_, _, payload, label) in enumerate(resolved):
            staged.append(
                _write_staged_output(
                    parent_fd,
                    payload,
                    label=label,
                    ordinal=ordinal,
                )
            )
        for (_, final_name, _, _), staged_identity in zip(
            resolved, staged, strict=True
        ):
            _publish_staged_output(
                parent_fd,
                staged_identity,
                final_name,
                published=published,
            )
        after = parent.stat(follow_symlinks=False)
        _reject(
            (after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino),
            "public release output directory changed during publication",
        )
        for (_, final_name, payload, label), identity in zip(
            resolved, published, strict=True
        ):
            final_stat = _entry_stat(parent_fd, final_name)
            _reject(
                final_stat is not None
                and (final_stat.st_dev, final_stat.st_ino) == (identity[1], identity[2])
                and stat.S_ISREG(final_stat.st_mode)
                and stat.S_IMODE(final_stat.st_mode) == 0o600
                and final_stat.st_nlink == 1
                and final_stat.st_size == len(payload),
                f"unexpected final {label} identity",
            )
        os.fsync(parent_fd)
    except BaseException as exc:
        cleanup_errors: list[str] = []
        if parent_fd is not None:
            for identity in reversed(published):
                cleanup_error = _unlink_owned_entry(parent_fd, identity)
                if cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)
            for identity in reversed(staged):
                cleanup_error = _unlink_owned_entry(parent_fd, identity)
                if cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise PublicReleaseAuditError(
                "public release cleanup incomplete: " + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _read_archive_once(path: Path) -> bytes:
    original = path.absolute()
    original_stat = original.lstat()
    _reject(not stat.S_ISLNK(original_stat.st_mode), "public archive path is a symlink")
    path = original.resolve(strict=True)
    before = path.stat(follow_symlinks=False)
    _reject(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and 0 < before.st_size <= MAX_ARCHIVE_BYTES,
        "existing public archive is not a bounded single-link regular file",
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _reject(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            "public archive changed before it was opened",
        )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            _reject(bool(chunk), "unexpected EOF while reading public archive")
            chunks.append(chunk)
            remaining -= len(chunk)
        _reject(os.read(descriptor, 1) == b"", "public archive grew while reading")
        after = os.fstat(descriptor)
        _reject(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "public archive changed while it was read",
        )
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def audit_existing_public_archive(path: Path) -> dict[str, Any]:
    archive_report = _audit_archive_bytes(_read_archive_once(path))
    return {
        "archive": {
            key: value
            for key, value in archive_report.items()
            if key not in {"public_paths", "excluded_paths"}
        },
        "audit_version": PUBLIC_RELEASE_AUDIT_VERSION,
        "classification": {
            "excluded_count": len(archive_report["excluded_paths"]),
            "included_count": len(archive_report["public_paths"]),
            "matches_embedded_closed_manifests": True,
        },
        "limitations": [
            "archive-only audit; source repository and commit identity were not authenticated",
            "unsigned local audit; no builder identity or provenance signature",
            "not an experiment reproduction or A24 formal result",
        ],
        "status": "PASSED",
    }


def build_and_audit_public_release(
    root: Path,
    *,
    output: Path | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    _reject(
        report_output is None or output is not None,
        "report output requires an archive output",
    )
    root = root.resolve(strict=True)
    _reject(root.is_dir(), "source root is not a directory")
    top_level = Path(
        _run_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    _reject(top_level == root, "--root must be the exact Git worktree root")
    _reject(
        _run_git(root, "status", "--porcelain=v1", "--untracked-files=all") == b"",
        "public release audit requires a clean worktree",
    )

    revision = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = (
        _run_git(root, "rev-parse", "--verify", f"{revision}^{{tree}}").decode().strip()
    )
    tracked, tree_listing_bytes = _tracked_paths(root, revision)
    include_paths = _parse_path_manifest(
        _run_git(root, "show", f"{revision}:{PUBLIC_INCLUDE_MANIFEST}"),
        label=PUBLIC_INCLUDE_MANIFEST,
    )
    exclude_paths = _parse_path_manifest(
        _run_git(root, "show", f"{revision}:{PUBLIC_EXCLUDE_MANIFEST}"),
        label=PUBLIC_EXCLUDE_MANIFEST,
    )
    executable_paths = _parse_path_manifest(
        _run_git(root, "show", f"{revision}:{PUBLIC_EXECUTABLE_MANIFEST}"),
        label=PUBLIC_EXECUTABLE_MANIFEST,
    )
    _reject(set(include_paths).isdisjoint(exclude_paths), "classification overlap")
    _reject(
        set(executable_paths) <= set(include_paths),
        "executable classification is outside the public include set",
    )
    _reject(
        set(tracked) == set(include_paths) | set(exclude_paths),
        "tracked paths do not exactly match the public include/exclude classification",
    )
    export_ignores = _parse_export_ignores(
        _run_git(root, "show", f"{revision}:.gitattributes")
    )
    _reject(export_ignores == exclude_paths, "committed export-ignore policy mismatch")
    for path, mode in tracked.items():
        expected_mode = "100755" if path in executable_paths else "100644"
        _reject(
            mode == expected_mode, f"tracked mode is not classified correctly: {path}"
        )

    archive_bytes = _run_git(
        root,
        "archive",
        "--format=tar",
        f"--prefix={ARCHIVE_PREFIX}/",
        revision,
        "--",
        *include_paths,
    )
    archive_report = _audit_archive_bytes(archive_bytes)
    _reject(
        archive_report["public_paths"] == include_paths
        and archive_report["excluded_paths"] == exclude_paths,
        "archive classification differs from the committed release policy",
    )
    _reject(
        archive_report["embedded_commit_claim"] == revision,
        "archive pax commit identity differs from the release revision",
    )
    _reject(
        _run_git(root, "status", "--porcelain=v1", "--untracked-files=all") == b""
        and _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        == revision,
        "source worktree or HEAD changed during the public release audit",
    )
    git_version = _run_git(root, "--version").decode("utf-8").strip()

    report = {
        "archive": {
            key: value
            for key, value in archive_report.items()
            if key not in {"public_paths", "excluded_paths"}
        },
        "audit_version": PUBLIC_RELEASE_AUDIT_VERSION,
        "classification": {
            "all_tracked_paths_classified": True,
            "excluded_count": len(exclude_paths),
            "included_count": len(include_paths),
            "tracked_count": len(tracked),
            "tracked_tree_bytes": tree_listing_bytes,
        },
        "limitations": [
            "explicit git archive only; Git history is not sanitized",
            "not a guarantee about GitHub auto-generated source archives",
            "unsigned local audit; no builder identity or provenance signature",
            "not a cross-machine reproducible-build result",
            "not an experiment reproduction or A24 formal result",
        ],
        "source": {
            "archive_built_from_expected_revision": True,
            "clean": True,
            "embedded_commit_matches_expected_revision": True,
            "git_version": git_version,
            "revision": revision,
            "tar_umask": "0022",
            "tree": tree,
        },
        "status": "PASSED",
    }
    outputs: list[tuple[Path, bytes, str]] = []
    if output is not None:
        outputs.append((output, archive_bytes, "public archive"))
    if report_output is not None:
        report_bytes = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        outputs.append((report_output, report_bytes, "public audit report"))
    if outputs:
        _write_private_outputs_once(outputs, root=root)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit the canonical sanitized MultiTown source tar."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--root",
        type=Path,
        help="Exact clean Git worktree root (default: current directory).",
    )
    source.add_argument(
        "--archive",
        type=Path,
        help="Audit an existing canonical public .tar without extracting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new .tar path outside the repository; never overwritten.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional JSON report paired with --output; never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.archive is not None:
            _reject(arguments.output is None, "--output cannot be used with --archive")
            _reject(
                arguments.report_output is None,
                "--report-output cannot be used with --archive",
            )
            report = audit_existing_public_archive(arguments.archive)
        else:
            report = build_and_audit_public_release(
                arguments.root or Path.cwd(),
                output=arguments.output,
                report_output=arguments.report_output,
            )
    except (OSError, PublicReleaseAuditError) as exc:
        print(f"public release audit rejected: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
