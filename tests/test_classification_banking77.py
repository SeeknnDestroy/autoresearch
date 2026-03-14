from pathlib import Path

from classification.banking77 import LABELS_FILENAME, load_examples, prepare_dataset


def _make_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = []
    for label in ("balance_not_updated_after_cheque_or_bank_transfer", "cash_withdrawal"):
        for idx in range(10):
            train_rows.append({"text": f"{label} train {idx}", "label": label})
    test_rows = []
    for label in ("balance_not_updated_after_cheque_or_bank_transfer", "cash_withdrawal"):
        for idx in range(3):
            test_rows.append({"text": f"{label} test {idx}", "label": label})
    return train_rows, test_rows


def test_prepare_dataset_is_deterministic(tmp_path: Path) -> None:
    train_rows, test_rows = _make_rows()
    dataset_dir = tmp_path / "banking77"

    first = prepare_dataset(
        dataset_dir,
        val_fraction=0.2,
        seed=1337,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    second = prepare_dataset(
        dataset_dir,
        val_fraction=0.2,
        seed=1337,
        force=True,
        train_rows=train_rows,
        test_rows=test_rows,
    )

    assert first == second
    assert first["train_examples"] == 16
    assert first["val_examples"] == 4
    assert first["test_examples"] == 6
    assert first["num_labels"] == 2

    labels_path = dataset_dir / LABELS_FILENAME
    assert labels_path.exists()

    train_examples = load_examples(dataset_dir, "train")
    val_examples = load_examples(dataset_dir, "val")
    assert len(train_examples) == 16
    assert len(val_examples) == 4
    assert {example.label for example in train_examples} == {example.label for example in val_examples}
