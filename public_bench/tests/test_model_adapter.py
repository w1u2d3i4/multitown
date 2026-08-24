from general_mas_bench.model_adapter import compact_messages, deterministic_request_seed


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
