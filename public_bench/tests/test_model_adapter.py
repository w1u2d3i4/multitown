from general_mas_bench.model_adapter import (
    ConservativeTokenBudgetAdapter,
    compact_messages,
    deterministic_request_seed,
)
from harness.agent_interface import AdapterResponse


def test_compaction_preserves_task_prompt_and_recent_turns() -> None:
    messages = [{"role": "user", "content": "task-spec"}]
    messages.extend(
        {"role": "user" if index % 2 else "assistant", "content": str(index) * 100}
        for index in range(1, 15)
    )

    effective, metadata = compact_messages(messages, char_budget=650)

    assert effective[0]["content"] == "task-spec"
    assert effective[-1]["content"] == "14" * 100
    assert "omitted" in effective[1]["content"]
    assert metadata["dropped_message_count"] > 0
    assert metadata["effective_message_count"] == len(effective)
    assert metadata["effective_message_chars"] <= 1_650


def test_short_history_is_unchanged() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "done"},
    ]

    effective, metadata = compact_messages(messages, char_budget=10)

    assert effective == messages
    assert metadata["dropped_message_count"] == 0


def test_request_seed_is_stable_across_method_runs() -> None:
    first = deterministic_request_seed(
        20260824, task_id="task-1", role="executor", request_index=2
    )
    assert first == deterministic_request_seed(
        20260824, task_id="task-1", role="executor", request_index=2
    )
    assert first != deterministic_request_seed(
        20260824, task_id="task-1", role="executor", request_index=3
    )
    assert 0 <= first <= 0x7FFFFFFF


class _FakeBudgetedAdapter:
    configured_max_tokens = 128

    def __init__(self) -> None:
        self.max_tokens = self.configured_max_tokens
        self.usage = 0
        self.calls = 0

    def set_request_max_tokens(self, value: int) -> None:
        self.max_tokens = value

    def generate_with_tools(self, messages, system_prompt, tools):
        self.calls += 1
        self.usage += 10 + self.max_tokens
        return AdapterResponse(text="ok")

    def get_usage(self):
        return {"total_tokens": self.usage}


def test_conservative_budget_blocks_request_before_upper_bound() -> None:
    inner = _FakeBudgetedAdapter()
    bounded = ConservativeTokenBudgetAdapter(
        inner,
        usage_provider=lambda: inner.usage,
        total_budget=100,
        template_overhead_tokens=80,
        minimum_completion_tokens=16,
    )
    response = bounded.generate_with_tools(
        [{"role": "user", "content": "task"}], "system", []
    )
    assert response.done is True
    assert inner.calls == 0
    assert bounded.events[0]["event"] == "request_blocked"


def test_conservative_budget_caps_completion_and_audits_usage() -> None:
    inner = _FakeBudgetedAdapter()
    bounded = ConservativeTokenBudgetAdapter(
        inner,
        usage_provider=lambda: inner.usage,
        total_budget=500,
        template_overhead_tokens=0,
        minimum_completion_tokens=16,
    )
    bounded.generate_with_tools(
        [{"role": "user", "content": "task"}], "system", []
    )
    assert inner.calls == 1
    assert 16 <= inner.max_tokens <= inner.configured_max_tokens
    assert inner.usage <= 500
