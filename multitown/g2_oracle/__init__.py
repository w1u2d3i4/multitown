"""Independent neutral-snapshot oracle for the G2 search vertical slice.

This package intentionally uses only the Python standard library. It must not
import the production environment, checker, evaluator, or behavioral probes.
"""

from .records_direct import (
    ORACLE_VERSION,
    build_records_direct_spec,
    evaluate_records_direct,
    oracle_spec_sha256,
)
from .resource_conflict import (
    FINAL_STATE_ONLY_MUTANT,
    RESOURCE_ORACLE_VERSION,
    build_resource_conflict_spec,
    evaluate_resource_conflict,
    resource_oracle_spec_sha256,
)
from .exact_replay import (
    EXACT_REPLAY_ORACLE_VERSION,
    build_exact_replay_spec,
    evaluate_exact_replay,
    exact_replay_spec_sha256,
)

__all__ = [
    "ORACLE_VERSION",
    "build_records_direct_spec",
    "evaluate_records_direct",
    "oracle_spec_sha256",
    "FINAL_STATE_ONLY_MUTANT",
    "RESOURCE_ORACLE_VERSION",
    "build_resource_conflict_spec",
    "evaluate_resource_conflict",
    "resource_oracle_spec_sha256",
    "EXACT_REPLAY_ORACLE_VERSION",
    "build_exact_replay_spec",
    "evaluate_exact_replay",
    "exact_replay_spec_sha256",
]
