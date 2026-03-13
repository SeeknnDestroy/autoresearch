from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from studio.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Autoresearch Studio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--studio-dir", default=".studio")
    args = parser.parse_args()

    app = create_app(Path(args.studio_dir).resolve())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
