"""Isolated, weights-only checkpoint inspection for the A24 verifier.

This module is intentionally separate from the training runner.  It accepts a
checkpoint that has already been copied into a verifier-owned private staging
directory, loads it with ``weights_only=True``, and emits only a bounded JSON
summary.  It never trains, evaluates episodes, or writes artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _apply_process_limits_from_environment() -> None:
    names = {
        "MULTITOWN_WORKER_MAX_AS": resource.RLIMIT_AS,
        "MULTITOWN_WORKER_MAX_CPU": resource.RLIMIT_CPU,
        "MULTITOWN_WORKER_MAX_FSIZE": resource.RLIMIT_FSIZE,
        "MULTITOWN_WORKER_MAX_NOFILE": resource.RLIMIT_NOFILE,
    }
    present = {name for name in names if name in os.environ}
    if not present:
        return
    if present != set(names):
        raise RuntimeError("checkpoint worker process-limit environment is incomplete")
    for name, limit_name in names.items():
        value = int(os.environ[name])
        if value <= 0:
            raise ValueError("checkpoint worker process limit must be positive")
        resource.setrlimit(limit_name, (value, value))


_apply_process_limits_from_environment()

import numpy as np
import torch
from torch import nn

WORKER_VERSION = "multitown-a24-weights-only-checkpoint-worker-v2"
_RUNTIME_KEYS = {
    "python",
    "numpy",
    "torch",
    "platform",
    "requested_threads",
    "torch_num_threads",
    "torch_num_interop_threads",
    "torch_deterministic_algorithms",
    "torch_deterministic_warn_only",
}
_PAYLOAD_KEYS = {
    "policy_version",
    "observation_size",
    "action_count",
    "hidden_size",
    "seed",
    "update",
    "ppo_config",
    "model_state",
    "optimizer_state",
    "pq1_primitives_version",
    "run_contract_sha256",
    "mechanism",
    "outer_fold",
    "training_log_sha256",
    "sample_sequence_sha256",
    "mode_sequence_sha256",
    "initial_model_sha256",
    "initial_optimizer_sha256",
    "final_model_sha256",
    "final_optimizer_sha256",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _all_finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is dict:
        return all(
            type(key) is str and _all_finite(item) for key, item in value.items()
        )
    if type(value) in {list, tuple}:
        return all(_all_finite(item) for item in value)
    return value is None or type(value) in {str, int, bool}


def strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {item}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid strict worker JSON: {label}") from exc
    if not _all_finite(value):
        raise RuntimeError(f"non-finite worker JSON value: {label}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _configure_frozen_runtime(runtime: Any) -> str:
    """Apply and verify the producer runtime settings that affect model bytes."""

    if type(runtime) is not dict or set(runtime) != _RUNTIME_KEYS:
        raise ValueError("checkpoint runtime contract schema changed")
    threads = runtime["torch_num_threads"]
    interop_threads = runtime["torch_num_interop_threads"]
    if (
        type(threads) is not int
        or not 1 <= threads <= 64
        or runtime["requested_threads"] != threads
        or type(interop_threads) is not int
        or not 1 <= interop_threads <= 256
        or runtime["python"] != sys.version
        or runtime["numpy"] != np.__version__
        or runtime["torch"] != torch.__version__
        or runtime["platform"] != platform.platform()
        or runtime["torch_deterministic_algorithms"] is not True
        or runtime["torch_deterministic_warn_only"] is not False
    ):
        raise ValueError("checkpoint runtime differs from the frozen producer")
    if torch.get_num_interop_threads() != interop_threads:
        raise ValueError("checkpoint interop thread runtime changed")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if (
        torch.get_num_threads() != threads
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError("checkpoint worker could not apply the frozen runtime")
    return hashlib.sha256(canonical_json_bytes(runtime)[:-1]).hexdigest()


def _observed_process_limits() -> dict[str, list[int]]:
    return {
        "address_space_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
        "cpu_seconds": list(resource.getrlimit(resource.RLIMIT_CPU)),
        "file_size_bytes": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
        "open_files": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
    }


def _isolation_state() -> dict[str, bool]:
    return {
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "isolated_mode": bool(sys.flags.isolated),
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
    }


class ActorCritic(nn.Module):
    """Verifier-local copy of the frozen A24 checkpoint architecture."""

    def __init__(self, observation_size: int, hidden_size: int, action_count: int):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.action_count = action_count
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        with torch.no_grad():
            self.actor.bias[7] = -1.5  # stop
            self.actor.bias[6] = -0.5  # human


def _hash_part(digest: Any, label: str, payload: bytes) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")


def _hash_value(digest: Any, label: str, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("checkpoint tensor is non-finite")
        array = tensor.numpy()
        little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        _hash_part(digest, f"{label}:tensor-dtype", little.dtype.str.encode("ascii"))
        _hash_part(
            digest,
            f"{label}:tensor-shape",
            json.dumps(list(little.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(digest, f"{label}:tensor-bytes", little.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise FloatingPointError("checkpoint array is non-finite")
        _hash_part(digest, f"{label}:array-dtype", array.dtype.str.encode("ascii"))
        _hash_part(
            digest,
            f"{label}:array-shape",
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(digest, f"{label}:array-bytes", array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        keys = sorted(value, key=lambda item: f"{type(item).__name__}:{item}")
        _hash_part(
            digest,
            f"{label}:mapping-keys",
            json.dumps(
                [f"{type(item).__name__}:{item}" for item in keys],
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for key in keys:
            _hash_value(digest, f"{label}.{type(key).__name__}:{key}", value[key])
        return
    if isinstance(value, (list, tuple)):
        _hash_part(
            digest,
            f"{label}:sequence-type",
            type(value).__name__.encode("ascii"),
        )
        for index, item in enumerate(value):
            _hash_value(digest, f"{label}[{index}]", item)
        return
    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not math.isfinite(value):
            raise FloatingPointError("checkpoint scalar is non-finite")
        _hash_part(
            digest,
            f"{label}:{type(value).__name__}",
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        )
        return
    raise TypeError(f"unsupported checkpoint value: {label}={type(value)!r}")


def _model_sha256(model: ActorCritic) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("model state must contain finite float32 tensors")
        little = np.ascontiguousarray(array, dtype="<f4")
        for payload in (
            key.encode("utf-8"),
            b"float32",
            json.dumps(list(little.shape), separators=(",", ":")).encode("ascii"),
            little.tobytes(order="C"),
        ):
            digest.update(payload)
            digest.update(b"\0")
    return digest.hexdigest()


def _optimizer_groups(
    optimizer: torch.optim.Optimizer, model: ActorCritic
) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    expected = [parameter for _, parameter in model.named_parameters()]
    observed: list[torch.nn.Parameter] = []
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        group_names: list[str] = []
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer parameter is foreign to the model")
            group_names.append(name)
            observed.append(parameter)
        groups.append(
            {
                **{key: value for key, value in group.items() if key != "params"},
                "params": group_names,
            }
        )
    if len(observed) != len(expected) or any(
        left is not right for left, right in zip(observed, expected, strict=True)
    ):
        raise ValueError("optimizer parameter order differs from the model")
    return groups


def _optimizer_sha256(optimizer: torch.optim.Optimizer, model: ActorCritic) -> str:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {
        names[id(parameter)]: values for parameter, values in optimizer.state.items()
    }
    digest = hashlib.sha256()
    _hash_value(
        digest,
        "optimizer",
        {
            "class": (
                f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
            ),
            "defaults": dict(optimizer.defaults),
            "param_groups": _optimizer_groups(optimizer, model),
            "state": state,
        },
    )
    return digest.hexdigest()


def _tensor_budget(value: Any, *, max_tensors: int, max_bytes: int) -> tuple[int, int]:
    tensors = 0
    total = 0
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if torch.is_tensor(current):
            tensors += 1
            total += current.numel() * current.element_size()
            if current.is_floating_point() and not bool(torch.isfinite(current).all()):
                raise FloatingPointError("checkpoint contains a non-finite tensor")
        elif isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                raise ValueError("checkpoint contains a cyclic mapping")
            seen.add(identity)
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen:
                raise ValueError("checkpoint contains a cyclic sequence")
            seen.add(identity)
            pending.extend(current)
        elif current is None or type(current) in {bool, int, float, str}:
            if type(current) is float and not math.isfinite(current):
                raise FloatingPointError("checkpoint contains a non-finite scalar")
        else:
            raise TypeError(f"checkpoint contains unsupported type: {type(current)!r}")
        if tensors > max_tensors or total > max_bytes:
            raise MemoryError("checkpoint tensor budget exceeded")
    return tensors, total


def inspect_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if set(expected) != {
        "fit",
        "ppo_config",
        "run_contract_sha256",
        "runtime",
        "max_tensors",
        "max_tensor_bytes",
    }:
        raise ValueError("checkpoint worker request schema changed")
    fit = expected["fit"]
    config = expected["ppo_config"]
    if type(fit) is not dict or type(config) is not dict:
        raise TypeError("checkpoint worker expected objects are malformed")
    runtime_sha256 = _configure_frozen_runtime(expected["runtime"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("checkpoint payload schema changed")
    tensors, tensor_bytes = _tensor_budget(
        payload,
        max_tensors=int(expected["max_tensors"]),
        max_bytes=int(expected["max_tensor_bytes"]),
    )
    exact = {
        "policy_version": "multitown-a24-cr-ppo-no-shield-policy-v1",
        "pq1_primitives_version": "multitown-pq1-rowwise-on-policy-primitives-v1",
        "mechanism": "cr-ppo-no-shield",
        "run_contract_sha256": expected["run_contract_sha256"],
        "update": config["updates"],
        "ppo_config": config,
        "observation_size": 47,
        "action_count": 8,
        "hidden_size": config["hidden_size"],
        "seed": fit["training_seed"],
        "outer_fold": fit["outer_fold"],
        "training_log_sha256": fit["training_log_sha256"],
        "sample_sequence_sha256": fit["sample_sequence_sha256"],
        "mode_sequence_sha256": fit["mode_sequence_sha256"],
        "initial_model_sha256": fit["initial_model_sha256"],
        "initial_optimizer_sha256": fit["initial_optimizer_sha256"],
        "final_model_sha256": fit["final_model_sha256"],
        "final_optimizer_sha256": fit["final_optimizer_sha256"],
    }
    if any(payload[key] != value for key, value in exact.items()):
        raise ValueError("checkpoint metadata differs from the fit receipt")
    random.seed(fit["training_seed"])
    np.random.seed(fit["training_seed"])
    torch.manual_seed(fit["training_seed"])
    initial_model = ActorCritic(47, config["hidden_size"], 8).cpu()
    initial_optimizer = torch.optim.Adam(
        initial_model.parameters(), lr=config["learning_rate"], eps=1e-5
    )
    initial_model_sha = _model_sha256(initial_model)
    initial_optimizer_sha = _optimizer_sha256(initial_optimizer, initial_model)
    if (
        initial_model_sha != fit["initial_model_sha256"]
        or initial_optimizer_sha != fit["initial_optimizer_sha256"]
    ):
        raise ValueError(
            "checkpoint initial state cannot be independently reconstructed"
        )
    model = ActorCritic(47, config["hidden_size"], 8).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], eps=1e-5
    )
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    final_model_sha = _model_sha256(model)
    final_optimizer_sha = _optimizer_sha256(optimizer, model)
    if (
        final_model_sha != fit["final_model_sha256"]
        or final_optimizer_sha != fit["final_optimizer_sha256"]
    ):
        raise ValueError("checkpoint final state digest mismatch")
    return {
        "schema_version": WORKER_VERSION,
        "checkpoint_payload_schema_valid": True,
        "weights_only_load": True,
        "runtime_sha256": runtime_sha256,
        "torch_num_threads": torch.get_num_threads(),
        "process_limits": _observed_process_limits(),
        "isolation": _isolation_state(),
        "initial_model_sha256": initial_model_sha,
        "initial_optimizer_sha256": initial_optimizer_sha,
        "final_model_sha256": final_model_sha,
        "final_optimizer_sha256": final_optimizer_sha,
        "tensor_count": tensors,
        "tensor_bytes": tensor_bytes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("checkpoint worker request exceeds the byte limit")
    request = strict_json_loads(raw.decode("utf-8"), label="checkpoint worker request")
    if type(request) is not dict:
        raise TypeError("checkpoint worker request is not an object")
    result = inspect_checkpoint(args.checkpoint, request)
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
