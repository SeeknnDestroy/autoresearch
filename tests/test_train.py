from collections import Counter
from pathlib import Path

import prepare
import train


def _context() -> prepare.ExperimentContext:
    examples = [
        prepare.Example(record_id="train-1", text="cash withdrawal charged twice", label="cash_withdrawal"),
        prepare.Example(record_id="train-2", text="cash withdrawal pending", label="cash_withdrawal"),
        prepare.Example(record_id="train-3", text="bank transfer pending", label="pending_transfer"),
        prepare.Example(record_id="train-4", text="bank transfer delayed", label="pending_transfer"),
    ]
    return prepare.ExperimentContext(labels=["cash_withdrawal", "pending_transfer"], train_examples=examples)


def test_build_request_shortlists_relevant_label() -> None:
    context = _context()
    example = prepare.Example(record_id="val-1", text="my cash withdrawal is missing", label="cash_withdrawal")

    request = train.build_request(example, context)

    assert "cash_withdrawal" in request.allowed_labels
    assert request.system_prompt
    assert '{"label": "one_of_the_allowed_labels"}' in request.user_prompt


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        target = "cash_withdrawal" if "cash withdrawal" in kwargs["prompt"] else "pending_transfer"
        return prepare.OllamaGeneration(raw_output=f'{{"label":"{target}"}}', latency_ms=50.0, model=kwargs["model"])


def test_run_experiment_prints_fixed_summary(tmp_path: Path, capsys) -> None:
    train_rows = [
        {"text": "cash withdrawal charged twice", "label": "cash_withdrawal"},
        {"text": "bank transfer pending", "label": "pending_transfer"},
    ]
    val_rows = [
        {"text": "cash withdrawal issue", "label": "cash_withdrawal"},
        {"text": "bank transfer issue", "label": "pending_transfer"},
    ]
    prepare.prepare_dataset(tmp_path / "dataset", force=True, train_rows=train_rows, val_rows=val_rows)

    def build_request(example: prepare.Example, context: prepare.ExperimentContext) -> prepare.InferenceRequest:
        allowed = context.labels
        return prepare.InferenceRequest(
            system_prompt="system",
            user_prompt=f"{example.text}\nAllowed labels: {allowed}",
            allowed_labels=allowed,
            options={"temperature": 0},
        )

    result = prepare.run_experiment(
        build_request,
        model_name="qwen3.5:0.8b",
        dataset_dir=tmp_path / "dataset",
        output_root=tmp_path / "results",
        client=_FakeClient(),
    )

    captured = capsys.readouterr().out
    assert "val_macro_f1:" in captured
    assert "val_accuracy:" in captured
    assert "invalid_rate:" in captured
    assert "avg_latency_ms:" in captured
    assert result["metrics"]["val_macro_f1"] == 1.0
