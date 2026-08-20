"""Generate A13 semantic MultiTown tasks with executable deterministic validators."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .masbench_routing import git_state, utc_now, write_json


SCHEMA_VERSION = "multitown-semantic-task-v2"
BANK_VERSION = "multitown-semantic-train-bank-v2"
FAMILIES = ("registry", "resource_dispatch", "telemetry", "safety_policy")
OPTION_LABELS = ("A", "B", "C", "D")
ABSTAIN_LABEL = "ABSTAIN"
DEFAULT_SEED = 20260813
CHOICE_PROMPT_PROTOCOL_VERSION = "multitown-semantic-choice-prompt-v3"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_id(
    split: str,
    seed: int,
    family: str,
    public_brief: str,
    options: tuple[str, ...],
    world_state_json: str,
    validator_ruleset: str,
) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "seed": seed,
        "family": family,
        "public_brief": public_brief,
        "options": list(options),
        "world_state": json.loads(world_state_json),
        "validator_ruleset": validator_ruleset,
    }
    digest = hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:16]
    return f"sem-{split}-{family}-{digest}"


@dataclass(frozen=True)
class SemanticTask:
    task_id: str
    split: str
    seed: int
    family: str
    public_brief: str
    options: tuple[str, ...]
    correct_option: int
    world_state_json: str
    validator_ruleset: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema: {self.schema_version}")
        if self.split != "train":
            raise ValueError("A13 v2 is a train-only task-design bank; evaluation splits are locked")
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported family: {self.family}")
        if len(self.options) != 4 or len(set(self.options)) != 4:
            raise ValueError("semantic tasks require four unique options")
        if not 0 <= self.correct_option < 4:
            raise ValueError("correct_option must be in [0, 3]")
        canonical_state = _canonical_json(json.loads(self.world_state_json))
        if canonical_state != self.world_state_json:
            raise ValueError("world_state_json must be canonical")
        expected_id = _content_id(
            self.split, self.seed, self.family, self.public_brief, self.options,
            self.world_state_json, self.validator_ruleset,
        )
        if self.task_id != expected_id:
            raise ValueError("task_id does not match content hash")
        if self.options[self.correct_option] != recompute_correct_action(self):
            raise ValueError("stored correct_option does not match executable validator")

    @property
    def world_state(self) -> dict[str, Any]:
        """Return a fresh copy; the frozen task never exposes mutable hidden state."""

        return json.loads(self.world_state_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "split": self.split,
            "seed": self.seed,
            "family": self.family,
            "public_brief": self.public_brief,
            "options": list(self.options),
            "metadata": {
                "template_cluster_id": self.family,
                "structural_signature": structural_signature(self),
                "world_seed": self.seed,
                "surface_seed": self.seed,
                "composition_signature": self.family,
            },
            "evaluator": {
                "correct_option": self.correct_option,
                "world_state": self.world_state,
                "validator_ruleset": self.validator_ruleset,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticTask":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema: {value.get('schema_version')}")
        evaluator = value["evaluator"]
        return cls(
            task_id=str(value["task_id"]), split=str(value["split"]),
            seed=int(value["seed"]), family=str(value["family"]),
            public_brief=str(value["public_brief"]),
            options=tuple(str(item) for item in value["options"]),
            correct_option=int(evaluator["correct_option"]),
            world_state_json=_canonical_json(evaluator["world_state"]),
            validator_ruleset=str(evaluator["validator_ruleset"]),
        )


def _shuffled_options(
    rng: random.Random, actions: list[str], correct_action: str,
) -> tuple[tuple[str, ...], int]:
    if len(actions) != 4 or len(set(actions)) != 4 or correct_action not in actions:
        raise ValueError("invalid semantic action set")
    values = list(actions)
    rng.shuffle(values)
    return tuple(values), values.index(correct_action)


GeneratorResult = tuple[str, list[str], str, dict[str, Any], str]


def _registry(rng: random.Random) -> GeneratorResult:
    sites = ("Riverside Market", "North Pier Workshop", "Juniper Food Hall", "Canal Depot")
    site = rng.choice(sites)
    status = rng.choice(("active", "expired", "revoked"))
    inspected = rng.choice((True, False))
    stale_status = rng.choice([item for item in ("active", "expired", "revoked") if item != status])
    actions = [
        "Authorize normal operation.",
        "Schedule the required safety inspection before opening.",
        "Request a permit renewal before any operation.",
        "Suspend the site and lock access pending appeal.",
    ]
    correct = (
        actions[3] if status == "revoked" else actions[2] if status == "expired"
        else actions[1] if not inspected else actions[0]
    )
    authority = rng.choice(("local", "central"))
    brief = (
        f"{site} asks the town to approve reopening today. The live authoritative "
        f"permit feed is the {authority} system. Choose the compliant next action."
    )
    state = {
        "site": site,
        "permit_status": status,
        "inspection_complete": inspected,
        "authority": authority,
        "stale_permit_status": stale_status,
        "stale_inspection_complete": not inspected,
    }
    return brief, actions, correct, state, "registry-priority-v2"


def _resource_dispatch(rng: random.Random) -> GeneratorResult:
    skills = ("electrical", "medical", "water", "structural")
    skill = rng.choice(skills)
    deadline = rng.choice((18, 22, 26, 30))
    crew_names = rng.sample(("Atlas", "Birch", "Cedar", "Delta", "Ember", "Fjord"), 4)
    correct_index = rng.randrange(4)
    rows = []
    failure_modes = ["wrong_skill", "unavailable", "late"]
    rng.shuffle(failure_modes)
    for index, name in enumerate(crew_names):
        if index == correct_index:
            row = {"name": name, "skill": skill, "available": True, "eta": deadline - rng.randint(2, 7)}
        else:
            failure = failure_modes[index if index < correct_index else index - 1]
            row = {
                "name": name,
                "skill": skill if failure != "wrong_skill" else rng.choice([x for x in skills if x != skill]),
                "available": failure != "unavailable",
                "eta": deadline + rng.randint(2, 8) if failure == "late" else deadline - rng.randint(1, 6),
            }
        rows.append(row)
    stale_rows = []
    for row in rows:
        stale_rows.append({
            "name": row["name"],
            "skill": rng.choice(skills),
            "available": rng.choice((True, False)),
            "eta": max(5, row["eta"] + rng.choice((-5, 4, 7))),
        })
    actions = [f"Dispatch crew {row['name']}." for row in rows]
    authority = rng.choice(("local", "central"))
    brief = (
        f"A {skill} incident must receive a qualified, available crew within {deadline} minutes. "
        f"The {authority} dispatch feed is live; the other feed is stale. Choose exactly one dispatch."
    )
    state = {
        "required_skill": skill,
        "deadline_minutes": deadline,
        "crews": rows,
        "stale_crews": stale_rows,
        "authority": authority,
    }
    return brief, actions, actions[correct_index], state, "dispatch-feasibility-v2"


def _telemetry(rng: random.Random) -> GeneratorResult:
    errors = ("AUTH_EXPIRED", "TIMEOUT", "CHECKSUM_MISMATCH")
    error = rng.choice(errors)
    attempts = rng.choice((0, 1, 2, 3))
    actions = [
        "Retry the same endpoint without changing credentials.",
        "Refresh the service credential, then retry once.",
        "Switch the request to the verified mirror endpoint.",
        "Quarantine the payload and open an integrity incident.",
    ]
    correct = (
        actions[1] if error == "AUTH_EXPIRED" else actions[3] if error == "CHECKSUM_MISMATCH"
        else actions[0] if attempts < 2 else actions[2]
    )
    job = f"sync-{rng.randrange(1000, 9999)}"
    authority = rng.choice(("local", "central"))
    stale_error = rng.choice([item for item in errors if item != error])
    stale_count = max(0, min(3, attempts + rng.choice((-1, 1))))
    brief = (
        f"Town data job {job} failed. The {authority} telemetry feed is live; the other feed is "
        "stale. Choose the runbook-compliant recovery action."
    )
    state = {
        "job": job,
        "terminal_error": error,
        "prior_timeout_count": attempts,
        "stale_error": stale_error,
        "stale_timeout_count": stale_count,
        "authority": authority,
    }
    return brief, actions, correct, state, "telemetry-recovery-v2"


def _safety(rng: random.Random) -> GeneratorResult:
    conditions = ("confirmed_leak", "unconfirmed_alarm", "false_alarm")
    condition = rng.choice(conditions)
    occupied = rng.choice((True, False))
    location = rng.choice(("East Tunnel", "Harbor Battery Room", "Old Mill Lab", "Metro Pump House"))
    actions = [
        "Continue normal operation and record the event.",
        "Deploy an instrument team to verify before changing operations.",
        "Isolate the location while keeping personnel outside the perimeter.",
        "Evacuate occupants and isolate the location immediately.",
    ]
    correct = (
        actions[3] if condition == "confirmed_leak" and occupied else
        actions[2] if condition == "confirmed_leak" else
        actions[1] if condition == "unconfirmed_alarm" else actions[0]
    )
    authority = rng.choice(("local", "central"))
    brief = (
        f"A safety alert was raised at {location}. The {authority} safety feed is live; the other "
        "feed is stale. Choose the policy-compliant response."
    )
    state = {
        "location": location,
        "alert_status": condition,
        "occupied": occupied,
        "stale_alert_status": rng.choice([item for item in conditions if item != condition]),
        "stale_occupied": not occupied,
        "authority": authority,
    }
    return brief, actions, correct, state, "safety-policy-v2"


GENERATORS: dict[str, Callable[[random.Random], GeneratorResult]] = {
    "registry": _registry,
    "resource_dispatch": _resource_dispatch,
    "telemetry": _telemetry,
    "safety_policy": _safety,
}


def generate_task(seed: int, split: str, family: str) -> SemanticTask:
    if split != "train":
        raise ValueError("A13 evaluation split generation remains locked")
    if family not in GENERATORS:
        raise ValueError(family)
    rng = random.Random(f"{SCHEMA_VERSION}:{split}:{family}:{seed}")
    brief, actions, correct, state, ruleset = GENERATORS[family](rng)
    options, correct_option = _shuffled_options(rng, actions, correct)
    state_json = _canonical_json(state)
    task_id = _content_id(split, seed, family, brief, options, state_json, ruleset)
    return SemanticTask(
        task_id=task_id, split=split, seed=seed, family=family,
        public_brief=brief, options=options, correct_option=correct_option,
        world_state_json=state_json, validator_ruleset=ruleset,
    )


def generate_bank(split: str, count: int, *, seed: int = DEFAULT_SEED) -> list[SemanticTask]:
    if count <= 0 or count % len(FAMILIES):
        raise ValueError(f"count must be a positive multiple of {len(FAMILIES)}")
    tasks = []
    per_family = count // len(FAMILIES)
    for family_index, family in enumerate(FAMILIES):
        for index in range(per_family):
            tasks.append(generate_task(seed + family_index * 100_000 + index, split, family))
    return sorted(tasks, key=lambda item: item.task_id)


def _policy_text(task: SemanticTask) -> str:
    if task.family == "registry":
        return (
            "Policy priority: revoked -> suspend; otherwise expired -> renew; otherwise incomplete "
            "inspection -> schedule inspection; active plus complete inspection -> authorize."
        )
    if task.family == "resource_dispatch":
        return "Dispatch requires matching certification, current availability, and ETA within the deadline."
    if task.family == "telemetry":
        return (
            "Runbook: AUTH_EXPIRED -> refresh credentials; CHECKSUM_MISMATCH -> quarantine; "
            "TIMEOUT -> retry when prior count is below 2, otherwise use verified mirror."
        )
    if task.family == "safety_policy":
        return (
            "Policy: confirmed leak + occupied -> evacuate and isolate; confirmed leak + empty -> "
            "isolate; unconfirmed alarm -> instrument verification; false alarm -> continue and record."
        )
    raise ValueError(task.family)


def role_context(task: SemanticTask, role: str) -> str:
    """Render a role view exclusively from immutable hidden state."""

    if role not in {"weak", "strong"}:
        raise ValueError(role)
    state = task.world_state
    live = state["authority"] == ("local" if role == "weak" else "central")
    source = "local specialist" if role == "weak" else "central generalist"
    status = "LIVE AUTHORITATIVE" if live else "STALE NON-AUTHORITATIVE"
    if task.family == "registry":
        permit = state["permit_status"] if live else state["stale_permit_status"]
        inspected = state["inspection_complete"] if live else state["stale_inspection_complete"]
        facts = (
            f"site={state['site']}; permit={permit}; inspection="
            f"{'complete' if inspected else 'incomplete'}"
        )
    elif task.family == "resource_dispatch":
        rows = state["crews"] if live else state["stale_crews"]
        facts = "\n".join(
            f"- {row['name']}: certification={row['skill']}; available="
            f"{'yes' if row['available'] else 'no'}; ETA={row['eta']} minutes"
            for row in rows
        )
    elif task.family == "telemetry":
        error = state["terminal_error"] if live else state["stale_error"]
        count = state["prior_timeout_count"] if live else state["stale_timeout_count"]
        facts = f"job={state['job']}; terminal_error={error}; prior_timeout_count={count}"
    elif task.family == "safety_policy":
        alert = state["alert_status"] if live else state["stale_alert_status"]
        occupied = state["occupied"] if live else state["stale_occupied"]
        facts = (
            f"location={state['location']}; alert_status={alert}; occupancy="
            f"{'occupied' if occupied else 'confirmed empty'}"
        )
    else:
        raise ValueError(task.family)
    return f"{source} feed status: {status}.\n{facts}.\n{_policy_text(task)}"


def render_worker_messages(task: SemanticTask, role: str) -> list[dict[str, str]]:
    role_text = (
        "local specialist using its assigned town tool"
        if role == "weak" else "central generalist using its assigned town systems"
    )
    option_text = "\n".join(
        f"{label}. {option}" for label, option in zip(OPTION_LABELS, task.options, strict=True)
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are a {role_text}. Select the one compliant action using only the provided "
                "brief, options, and role feed. If the feed is LIVE AUTHORITATIVE, return the "
                'selected label as {"option":"A"}. If the feed is STALE NON-AUTHORITATIVE, return '
                '{"option":"ABSTAIN"}; do not guess from stale facts. Return only that JSON object.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task brief:\n{task.public_brief}\n\nOptions:\n{option_text}\n\n"
                f"Role feed:\n{role_context(task, role)}"
            ),
        },
    ]


def parse_option(content: str) -> tuple[int | None, str | None]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
        option = value.get("option") if isinstance(value, dict) else None
        if option not in OPTION_LABELS or set(value) != {"option"}:
            raise ValueError("response must contain only option A, B, C, or D")
        return OPTION_LABELS.index(option), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parse_decision(content: str) -> tuple[int | None, bool, str | None]:
    """Parse a strict A-D choice or an explicit abstention."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
        option = value.get("option") if isinstance(value, dict) else None
        if set(value) != {"option"} or option not in (*OPTION_LABELS, ABSTAIN_LABEL):
            raise ValueError("response must contain only option A, B, C, D, or ABSTAIN")
        if option == ABSTAIN_LABEL:
            return None, True, None
        return OPTION_LABELS.index(option), False, None
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


def recompute_correct_action(task: SemanticTask) -> str:
    """Execute the named validator against hidden world state, ignoring its stored label."""

    state = task.world_state
    if task.validator_ruleset == "registry-priority-v2":
        if state["permit_status"] == "revoked":
            return "Suspend the site and lock access pending appeal."
        if state["permit_status"] == "expired":
            return "Request a permit renewal before any operation."
        if not state["inspection_complete"]:
            return "Schedule the required safety inspection before opening."
        return "Authorize normal operation."
    if task.validator_ruleset == "dispatch-feasibility-v2":
        eligible = [
            row for row in state["crews"]
            if row["skill"] == state["required_skill"] and row["available"]
            and row["eta"] <= state["deadline_minutes"]
        ]
        if len(eligible) != 1:
            raise ValueError(f"dispatch validator expected one eligible crew, found {len(eligible)}")
        return f"Dispatch crew {eligible[0]['name']}."
    if task.validator_ruleset == "telemetry-recovery-v2":
        if state["terminal_error"] == "AUTH_EXPIRED":
            return "Refresh the service credential, then retry once."
        if state["terminal_error"] == "CHECKSUM_MISMATCH":
            return "Quarantine the payload and open an integrity incident."
        if state["terminal_error"] == "TIMEOUT" and state["prior_timeout_count"] < 2:
            return "Retry the same endpoint without changing credentials."
        if state["terminal_error"] == "TIMEOUT":
            return "Switch the request to the verified mirror endpoint."
        raise ValueError(f"unknown terminal error: {state['terminal_error']}")
    if task.validator_ruleset == "safety-policy-v2":
        if state["alert_status"] == "confirmed_leak" and state["occupied"]:
            return "Evacuate occupants and isolate the location immediately."
        if state["alert_status"] == "confirmed_leak":
            return "Isolate the location while keeping personnel outside the perimeter."
        if state["alert_status"] == "unconfirmed_alarm":
            return "Deploy an instrument team to verify before changing operations."
        if state["alert_status"] == "false_alarm":
            return "Continue normal operation and record the event."
        raise ValueError(f"unknown alert status: {state['alert_status']}")
    raise ValueError(f"unknown validator ruleset: {task.validator_ruleset}")


def recompute_correct_option(task: SemanticTask) -> int:
    action = recompute_correct_action(task)
    if task.options.count(action) != 1:
        raise ValueError("validator action is not represented exactly once")
    return task.options.index(action)


def verify_option(task: SemanticTask, option: int) -> bool:
    return option == recompute_correct_option(task)


def review_current_candidate(
    task: SemanticTask, *, current_candidate: int, review_count: int,
) -> dict[str, Any]:
    """Single-use pass/fail review; the environment owns and increments review_count."""

    if review_count != 0:
        raise RuntimeError("review is single-use for each task")
    if not 0 <= current_candidate < len(task.options):
        raise ValueError("current_candidate is outside the option set")
    return {
        "verifier": "single-use-deterministic-state-checker-v2",
        "candidate_sha256": hashlib.sha256(task.options[current_candidate].encode()).hexdigest(),
        "passed": verify_option(task, current_candidate),
        "next_review_count": 1,
    }


def hidden_field_name_audit(tasks: list[SemanticTask]) -> dict[str, Any]:
    """Catch direct hidden-field names; this is not a semantic shortcut audit."""

    banned = ("correct_option", "world_state", "validator_ruleset", "evaluator")
    findings = []
    for task in tasks:
        for role in ("weak", "strong"):
            rendered = json.dumps(render_worker_messages(task, role), ensure_ascii=False).lower()
            leaked = [name for name in banned if name in rendered]
            if leaked:
                findings.append({"task_id": task.task_id, "role": role, "fields": leaked})
    return {
        "schema_version": "multitown-hidden-field-name-audit-v1",
        "scope": "direct hidden field names only; does not prove absence of semantic shortcuts",
        "tasks": len(tasks), "passed": not findings, "findings": findings,
    }


def structural_signature(task: SemanticTask) -> str:
    state = task.world_state
    if task.family == "registry":
        value = [task.family, state["authority"], state["permit_status"], state["inspection_complete"]]
    elif task.family == "resource_dispatch":
        value = [
            task.family, state["authority"], state["required_skill"], state["deadline_minutes"],
            [[row["skill"], row["available"], row["eta"] - state["deadline_minutes"]] for row in state["crews"]],
        ]
    elif task.family == "telemetry":
        value = [task.family, state["authority"], state["terminal_error"], state["prior_timeout_count"]]
    elif task.family == "safety_policy":
        value = [task.family, state["authority"], state["alert_status"], state["occupied"]]
    else:
        raise ValueError(task.family)
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def read_bank(path: Path) -> list[SemanticTask]:
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(SemanticTask.from_dict(json.loads(line)))
    if not tasks:
        raise ValueError(f"no semantic tasks in {path}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate semantic task ids")
    return tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_bank(output_dir: Path, *, split: str, count: int, seed: int) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    generator_sha256 = _sha256(Path(__file__))
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bank: {output_dir}")
    tasks = generate_bank(split, count, seed=seed)
    hidden_audit = hidden_field_name_audit(tasks)
    if not hidden_audit["passed"]:
        raise RuntimeError("hidden-field-name audit failed")
    signatures = Counter(structural_signature(task) for task in tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / f"{split}.jsonl"
    with bank_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output_dir / "hidden-field-name-audit.json", hidden_audit)
    manifest = {
        "schema_version": BANK_VERSION,
        "created_at_utc": utc_now(),
        "evaluation_status": "A13 train-only atomic task-design bank; evaluation banks do not exist",
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "generator_sha256": generator_sha256,
        "python": platform.python_version(),
        "split": split,
        "seed": seed,
        "task_count": len(tasks),
        "family_counts": dict(sorted(Counter(task.family for task in tasks).items())),
        "authority_counts": dict(sorted(Counter(task.world_state["authority"] for task in tasks).items())),
        "correct_option_counts": dict(sorted(Counter(OPTION_LABELS[task.correct_option] for task in tasks).items())),
        "unique_structural_signatures": len(signatures),
        "max_tasks_per_structural_signature": max(signatures.values()),
        "worker_roles": {
            "weak": "local specialist; live or stale feed varies within every family",
            "strong": "central generalist; live or stale feed varies within every family",
        },
        "reviewer": "single-use candidate-bound deterministic pass/fail checker",
        "bank_sha256": _sha256(bank_path),
        "hidden_field_name_audit_sha256": _sha256(output_dir / "hidden-field-name-audit.json"),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = freeze_bank(args.output_dir, split=args.split, count=args.count, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
