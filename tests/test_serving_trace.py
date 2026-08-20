import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from multitown.serving_trace import (
    export_a8_trace,
    validate_nano_replay,
    validate_trace,
)
from multitown.nanovllm_replay import summarize


class ServingTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        start = datetime(2026, 8, 10, tzinfo=timezone.utc)
        phases = [
            ("initial_attempt", "weak_initial_solver", "weak", "alpha"),
            ("selective_weak_delegation", "weak_specialist", "weak", "beta"),
            ("strong_specialist_escalation", "strong_specialist", "strong", "beta"),
            ("independent_review", "independent_reviewer", "strong", "alpha"),
        ]
        request_rows = []
        phase_trace = []
        for index, (phase, role, tier, candidate) in enumerate(phases):
            request_rows.append({
                "timestamp_utc": (start + timedelta(seconds=index)).isoformat(),
                "scenario_id": "private-scenario-007",
                "phase": phase,
                "role": role,
                "model_tier": tier,
                "model": "private-model-alias",
                "endpoint": "http://127.0.0.1:8001/v1",
                "inference_seed": 100 + index,
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 64,
                "action": candidate,
                "latency_s": 0.5 + index,
                "ttft_s": 0.1 + index,
                "prompt_tokens": 20 + index,
                "completion_tokens": 3,
                "messages": [
                    {"role": "system", "content": "shared system policy"},
                    {"role": "user", "content": f"private task turn {index}"},
                ],
            })
            phase_trace.append({
                "phase": phase,
                "role": role,
                "action": candidate,
                "validation": {
                    "parse_valid": True,
                    "hard_constraints_pass": index > 1,
                    "issue_codes": [] if index > 1 else ["needs_review"],
                    "checked_action": candidate,
                },
            })
        decision = {
            "timestamp_utc": (start + timedelta(seconds=5)).isoformat(),
            "scenario_id": "private-scenario-007",
            "request_count": len(request_rows),
            "phase_trace": phase_trace,
            "human_escalation_required": True,
            "decision_latency_s": 5.0,
            "final_validation": {
                "parse_valid": True,
                "hard_constraints_pass": False,
                "issue_codes": ["human_required"],
                "checked_action": "alpha",
            },
            "route": "human_escalation_required",
            "oracle_action": "beta",
            "correct": False,
        }
        self.requests = self.root / "requests.jsonl"
        self.decisions = self.root / "decisions.jsonl"
        self.requests.write_text(
            "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
        )
        self.decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    def test_export_is_sanitized_and_valid(self) -> None:
        trace = self.root / "trace.jsonl"
        nano = self.root / "nano.jsonl"
        sample_trace = self.root / "sample-trace.jsonl"
        sample_nano = self.root / "sample-nano.jsonl"
        manifest = export_a8_trace(
            requests_path=self.requests,
            decisions_path=self.decisions,
            trace_path=trace,
            nano_path=nano,
            manifest_path=self.root / "manifest.json",
            sample_trace_path=sample_trace,
            sample_nano_path=sample_nano,
            sample_sessions=1,
            salt=b"fixture salt",
        )
        self.assertEqual(manifest["sessions"], 1)
        self.assertEqual(manifest["request_events"], 4)
        rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [row["action"] for row in rows],
            ["delegate", "delegate", "escalate", "review", "human"],
        )
        serialized = trace.read_text(encoding="utf-8") + nano.read_text(encoding="utf-8")
        for secret in (
            "private-scenario-007",
            "127.0.0.1",
            "oracle_action",
            "private-model-alias",
        ):
            self.assertNotIn(secret, serialized)
        self.assertTrue(validate_trace(trace)["passed"])
        self.assertTrue(validate_trace(sample_trace)["passed"])
        self.assertTrue(validate_nano_replay(nano)["passed"])
        self.assertTrue(validate_nano_replay(sample_nano)["passed"])
        replay_rows = [json.loads(line) for line in nano.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summarize(replay_rows)["requests"], 4)

    def test_validator_rejects_endpoint_leak(self) -> None:
        trace = self.root / "trace.jsonl"
        export_a8_trace(
            requests_path=self.requests,
            decisions_path=self.decisions,
            trace_path=trace,
            nano_path=self.root / "nano.jsonl",
            manifest_path=self.root / "manifest.json",
            salt=b"fixture salt",
        )
        rows = trace.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        first["request"]["endpoint"] = "http://127.0.0.1:8001/v1"
        rows[0] = json.dumps(first)
        trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = validate_trace(trace)
        self.assertFalse(result["passed"])
        self.assertTrue(any("forbidden key" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
