# autoresearch

This fork turns `autoresearch` into a **Karpathy-style autonomous classification repo**.

The contract is intentionally small:

- `prepare.py` is fixed and read-only
- `train.py` is the only file the agent edits during experiments
- `program.md` is the human-authored operating manual

The task is **Banking77 intent classification** with **Ollama `qwen3.5:0.8b`**. Every run evaluates on the same fixed validation set and prints the same summary metrics, so the agent can keep or discard each experiment exactly like the original repo.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Prepare the fixed dataset/evaluator and verify Ollama readiness
uv run prepare.py

# 3. Run one full evaluation experiment
uv run train.py
```

## How It Works

Only three files matter for the autonomous loop:

- `prepare.py` — fixed Banking77 prep, Ollama runtime, parsing, metrics, and experiment summary
- `train.py` — the single editable experiment file
- `program.md` — instructions for the autonomous experiment loop

The agent should:

1. create a fresh `autoresearch/<tag>` branch
2. initialize untracked `results.tsv`
3. establish a baseline with `uv run train.py > run.log 2>&1`
4. edit only `train.py`
5. commit every experiment
6. rerun the experiment
7. keep improved runs
8. discard worse runs by resetting to the prior kept commit
9. continue until interrupted

## Fixed Metrics

Each run prints:

```text
---
val_macro_f1:     0.123456
val_accuracy:     0.234567
invalid_rate:     0.012345
avg_latency_ms:   987.6
total_examples:   3080
total_seconds:    1234.5
```

Comparison rule:

1. higher `val_macro_f1`
2. lower `invalid_rate`
3. higher `val_accuracy`
4. lower `avg_latency_ms`

## Logging

`results.tsv` must stay untracked and use this schema:

```text
commit	val_macro_f1	val_accuracy	invalid_rate	avg_latency_ms	status	description
```

## Legacy Code

The old pretraining implementation and the earlier additive classification prototype are archived under `legacy/`. They are not part of the main workflow for the fresh autoresearch session.
