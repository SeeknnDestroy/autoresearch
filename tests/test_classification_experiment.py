from pathlib import Path

from classification.experiment import decide_status, run_experiment_loop


def test_decide_status_only_keeps_improvements() -> None:
    assert decide_status(0.8, []) == "keep"
    assert decide_status(0.8, [{"val_macro_f1": "0.75", "status": "keep"}]) == "keep"
    assert decide_status(0.8, [{"val_macro_f1": "0.80", "status": "keep"}]) == "discard"
    assert decide_status(0.8, [{"val_macro_f1": "0.90", "status": "keep"}]) == "discard"


def test_run_experiment_loop_logs_row(monkeypatch, tmp_path: Path) -> None:
    def fake_evaluate_split(*args, **kwargs):
        output_dir = tmp_path / "eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.jsonl"
        metrics_path = output_dir / "metrics.json"
        predictions_path.write_text("", encoding="utf-8")
        metrics_path.write_text("{}", encoding="utf-8")
        return {
            "metrics": {
                "macro_f1": 0.75,
                "accuracy": 0.80,
                "invalid_rate": 0.10,
                "avg_latency_ms": 50.0,
            },
            "predictions_path": predictions_path,
            "metrics_path": metrics_path,
            "output_dir": output_dir,
        }

    monkeypatch.setattr("classification.experiment.evaluate_split", fake_evaluate_split)
    monkeypatch.setattr("classification.experiment._git_short_commit", lambda: "abc1234")

    results_path = tmp_path / "results.tsv"
    result = run_experiment_loop(results_path=results_path, description="baseline")

    assert result["status"] == "keep"
    content = results_path.read_text(encoding="utf-8")
    assert "abc1234\t0.750000\t0.800000\t0.100000\t50.0\tkeep\tbaseline" in content
