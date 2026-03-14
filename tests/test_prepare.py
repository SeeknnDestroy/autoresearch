import json
from pathlib import Path

import prepare


def _dataset_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [
        {"text": "cash withdrawal charged twice", "label": "cash_withdrawal"},
        {"text": "cash withdrawal pending", "label": "cash_withdrawal"},
        {"text": "bank transfer pending", "label": "pending_transfer"},
        {"text": "bank transfer delayed", "label": "pending_transfer"},
    ]
    val_rows = [
        {"text": "my cash withdrawal is missing", "label": "cash_withdrawal"},
        {"text": "my transfer is still pending", "label": "pending_transfer"},
    ]
    return train_rows, val_rows


def test_prepare_dataset_writes_fixed_train_and_val(tmp_path: Path) -> None:
    train_rows, val_rows = _dataset_rows()

    metadata = prepare.prepare_dataset(
        tmp_path,
        force=True,
        train_rows=train_rows,
        val_rows=val_rows,
    )

    assert metadata["train_examples"] == 4
    assert metadata["val_examples"] == 2
    assert metadata["num_labels"] == 2
    assert len(prepare.load_examples("train", tmp_path)) == 4
    assert len(prepare.load_examples("val", tmp_path)) == 2
    assert prepare.load_labels(tmp_path) == ["cash_withdrawal", "pending_transfer"]


def test_parse_prediction_accepts_humanized_label() -> None:
    label, is_valid, error = prepare.parse_prediction(
        '{"label":"cash withdrawal"}',
        ["cash_withdrawal", "pending_transfer"],
    )

    assert label == "cash_withdrawal"
    assert is_valid is True
    assert error is None


def test_compute_metrics_handles_invalid_predictions() -> None:
    metrics = prepare.compute_metrics(
        [
            {
                "gold_label": "cash_withdrawal",
                "pred_label": "cash_withdrawal",
                "is_valid": True,
                "latency_ms": 100.0,
            },
            {
                "gold_label": "pending_transfer",
                "pred_label": None,
                "is_valid": False,
                "latency_ms": 200.0,
            },
        ],
        ["cash_withdrawal", "pending_transfer"],
    )

    assert metrics["val_accuracy"] == 0.5
    assert metrics["invalid_rate"] == 0.5
    assert metrics["avg_latency_ms"] == 150.0
    assert metrics["val_macro_f1"] == 0.5


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "timeout": timeout, **kwargs})
        return _FakeResponse({"response": json.dumps({"label": "cash_withdrawal"}), "total_duration": 1_000_000})


def test_ollama_client_disables_thinking_by_default() -> None:
    client = prepare.OllamaClient(session=_FakeSession())
    generation = client.generate_json(
        model="qwen3.5:0.8b",
        system_prompt="system",
        prompt="prompt",
        options={"temperature": 0},
    )

    payload = client.session.calls[0]["json"]
    assert payload["think"] is False
    assert generation.raw_output == '{"label": "cash_withdrawal"}'
