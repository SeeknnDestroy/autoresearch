## Repo Memory

- Purpose: minimal autonomous LLM pretraining research harness; an agent iterates on `train.py` while `prepare.py` and the evaluation metric stay fixed.
- Core workflow: human edits `program.md` to define the research org/instructions; agent edits `train.py`, runs 5-minute experiments, and keeps only changes that improve `val_bpb`.
- Repo shape: this is a compact experiment scaffold, not an application framework or reusable product library.
- User preference: user is primarily an applied/agentic AI engineer, so explanations and reuse suggestions should focus on orchestration, evaluation loops, and engineering workflow rather than model-training depth.
- Fork direction: add `Autoresearch Studio` as an additive Mac-first local lab instead of rewriting the original CUDA-centric harness.
- Studio architecture: Python-served web app in `studio/` with FastAPI, SSE updates, persisted `.studio/` session storage, and a cheap TinyStories-style char-RNN backend for local baseline/candidate runs.
- Frontend direction: editorial lab notebook + research cockpit aesthetic, with browser QA covered by Playwright.
- Packaging note: `torch` now falls back to the default PyPI source on macOS while keeping the `pytorch-cu128` index for non-Darwin platforms.
- UX direction update: the Studio now leans into a stage-like “research cabaret” feel with live stage headlines, scorecards, mutation atlas, and run spotlight rather than a plain dashboard.
- UX correction: the cabaret/dashboard view is no longer the default mental model. The correct default is Guided Play Mode for nontechnical users, with the dense dashboard demoted behind an Advanced Lab toggle.
- Teaching rule: Play Mode must explain the repo goal, the four-step experiment loop, the latest verdict, the last change, and the three key repo files in plain language before any raw metrics or jargon show up.
- Reliability lesson: session persistence must use atomic writes plus a tiny retry on reads; Playwright caught a real race where the UI could read `session.json` mid-write.
- Experiment behavior update: default sessions now run 5 local experiments and the proposer avoids repeating the same field in back-to-back candidate moves, which makes sessions more interesting to watch.
