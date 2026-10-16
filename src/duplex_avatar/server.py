from __future__ import annotations

import argparse

import uvicorn

from duplex_avatar.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["mock", "stableavatar", "stableavatar-1.3b", "h3-diffsynth"], default=None)
    parser.add_argument("--brain", choices=["mock", "remote", "local", "minicpm-o"], default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    app = create_app(args.config, backend_override=args.backend, brain_override=args.brain)
    uvicorn.run(app, host=args.host or "0.0.0.0", port=args.port or 8000)


if __name__ == "__main__":
    main()
