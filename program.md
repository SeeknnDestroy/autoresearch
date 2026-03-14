# autoresearch

This repo is an experiment to have the LLM do its own classification research.

## Setup

To set up a new run:

1. Agree on a run tag based on today's date, for example `mar14`.
2. Create a fresh branch: `git checkout -b autoresearch/<tag>` from current `master`.
3. Read the only in-scope files for the loop:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
4. Verify the environment:
   - `uv run prepare.py`
   - this must confirm that Banking77 is cached and `qwen3.5:0.8b` is available in Ollama
5. Initialize `results.tsv` with this exact tab-separated header and leave it untracked:

```
commit	val_macro_f1	val_accuracy	invalid_rate	avg_latency_ms	status	description
```

6. Confirm setup looks good and begin the loop.

## Rules

Only three files matter for the experiment loop:

- `prepare.py` — fixed dataset prep, runtime utilities, and evaluator. Read-only.
- `train.py` — the only file you may modify during experiments.
- `program.md` — the instructions you are following. Read-only.

What you CAN do:

- Modify `train.py`.
- Run git commands needed for the keep/discard loop.
- Run `uv run train.py > run.log 2>&1`.
- Read `run.log`.
- Append rows to the untracked `results.tsv`.

What you CANNOT do:

- Modify `prepare.py`.
- Modify the evaluation harness, parsing rules, metric definitions, or tie-break rules.
- Add dependencies or modify `pyproject.toml`.
- Modify `README.md`, `program.md`, tests, or any file other than `train.py`.

## Goal

Get the highest `val_macro_f1` on the fixed validation set.

Tie-break rules, in order:

1. lower `invalid_rate`
2. higher `val_accuracy`
3. lower `avg_latency_ms`

All comparisons must use the metrics printed by `uv run train.py`.

## Output Format

Every run prints a fixed summary like this:

```text
---
val_macro_f1:     0.123456
val_accuracy:     0.234567
invalid_rate:     0.012345
avg_latency_ms:   987.6
total_examples:   3080
total_seconds:    1234.5
```

You can extract the comparison metrics from the log with:

```bash
grep "^val_macro_f1:\|^val_accuracy:\|^invalid_rate:\|^avg_latency_ms:" run.log
```

If the grep output is empty, the run crashed. Read the stack trace with:

```bash
tail -n 50 run.log
```

## Logging Results

Append every completed experiment to `results.tsv` using these columns:

1. short git commit hash
2. `val_macro_f1`
3. `val_accuracy`
4. `invalid_rate`
5. `avg_latency_ms`
6. `status` — one of `keep`, `discard`, or `crash`
7. short description of the experiment

Example:

```text
commit	val_macro_f1	val_accuracy	invalid_rate	avg_latency_ms	status	description
a1b2c3d	0.412300	0.556100	0.031200	932.4	keep	baseline
b2c3d4e	0.427800	0.571400	0.028900	905.7	keep	increase shortlist to 10
c3d4e5f	0.421100	0.565900	0.026100	901.3	discard	remove few-shot examples
d4e5f6g	0.000000	0.000000	1.000000	0.0	crash	bad JSON repair logic
```

## Exact Loop

LOOP FOREVER:

1. Inspect the current branch and current kept commit.
2. If no baseline exists yet, run the baseline first without modifying `train.py`.
3. Otherwise choose one new experimental idea and edit only `train.py`.
4. Commit the change.
5. Run the experiment:

```bash
uv run train.py > run.log 2>&1
```

6. Extract the metrics from `run.log`.
7. Append one row to `results.tsv`.
8. If the run improved according to the fixed comparison rules, keep the commit and advance from it.
9. If the run is worse or tied after tie-breakers, reset back to the previous kept commit.
10. Continue immediately with the next idea.

If a run crashes:

- use judgment to decide if it is a trivial fix
- if it is not a trivial fix, log it as `crash`, reset to the previous kept commit, and move on

## Autonomy

Do not pause to ask whether you should continue once the loop starts.

Do not ask whether this is a good stopping point.

Do not stop because you found one improvement.

Keep running until the human explicitly interrupts you.
