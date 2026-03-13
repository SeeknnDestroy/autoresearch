from pathlib import Path

from studio.tasks.tinystories_local import TinyStoriesLocalBackend


def test_backend_baseline_candidate_cycle(tmp_path: Path) -> None:
    backend = TinyStoriesLocalBackend(
        tmp_path,
        device_preference="cpu",
        recipe_overrides={
            "sequence_len": 24,
            "batch_size": 8,
            "embedding_dim": 24,
            "hidden_dim": 48,
            "steps": 6,
            "eval_batches": 2,
            "sample_length": 50,
        },
    )

    current_recipe = backend.prepare()
    baseline_dir = tmp_path / "baseline"
    baseline = backend.run_baseline(run_dir=baseline_dir)

    assert baseline.metrics["val_bpb"] > 0
    assert baseline.sample_text
    assert backend.extract_metrics(baseline_dir)["val_bpb"] == baseline.metrics["val_bpb"]
    assert backend.generate_samples(baseline_dir) == baseline.sample_text

    change_spec = backend.propose_change([], current_recipe)
    candidate_dir = tmp_path / "candidate"
    candidate = backend.run_candidate(change_spec, run_dir=candidate_dir)

    assert candidate.metrics["train_loss"] > 0
    assert change_spec["field"] in backend.recipe_diff(current_recipe, candidate.recipe)

    restored = backend.rollback(baseline.recipe_sha)
    assert restored == current_recipe
