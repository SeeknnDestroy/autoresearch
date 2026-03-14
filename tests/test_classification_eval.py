from classification.eval import parse_prediction
from classification.metrics import compute_metrics
from classification.profile import resolve_default_profile_path


def test_parse_prediction_accepts_humanized_labels() -> None:
    pred_label, is_valid, error = parse_prediction(
        '{"label":"cash withdrawal"}',
        labels=["cash_withdrawal", "beneficiary_not_allowed"],
    )

    assert pred_label == "cash_withdrawal"
    assert is_valid is True
    assert error is None


def test_parse_prediction_rejects_invalid_json() -> None:
    pred_label, is_valid, error = parse_prediction(
        "not-json",
        labels=["cash_withdrawal"],
    )

    assert pred_label is None
    assert is_valid is False
    assert error.startswith("invalid_json:")


def test_compute_metrics_handles_invalid_predictions() -> None:
    metrics = compute_metrics(
        [
            {
                "gold_label": "cash_withdrawal",
                "pred_label": "cash_withdrawal",
                "is_valid": True,
                "latency_ms": 100.0,
            },
            {
                "gold_label": "beneficiary_not_allowed",
                "pred_label": None,
                "is_valid": False,
                "latency_ms": 200.0,
            },
        ],
        labels=["cash_withdrawal", "beneficiary_not_allowed"],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["invalid_rate"] == 0.5
    assert metrics["avg_latency_ms"] == 150.0
    assert metrics["macro_f1"] == 0.5


def test_default_profile_path_prefers_repo_cwd(tmp_path, monkeypatch) -> None:
    profile_path = tmp_path / "classification_profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_default_profile_path() == profile_path
