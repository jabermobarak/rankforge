from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import export_csv, from_csv, synthetic
from .train import train


def main():
    parser = argparse.ArgumentParser(description="RankForge training and serving")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, default=Path("artifacts"))
        command.add_argument("--epochs", type=int, default=30)
        command.add_argument("--seed", type=int, default=42)
        if name == "demo":
            command.add_argument("--users", type=int, default=400)
            command.add_argument("--items", type=int, default=320)
            command.add_argument("--export-csv", type=Path)
        else:
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--train-end", type=int, required=True)
            command.add_argument("--validation-end", type=int, required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("rankforge.api:create_app", factory=True, host=args.host, port=args.port)
        return
    data = (
        synthetic(args.seed, args.users, args.items)
        if args.command == "demo"
        else from_csv(args.data, args.train_end, args.validation_end)
    )
    if args.command == "demo" and args.export_csv:
        export_csv(data, args.export_csv)
    report = train(data, args.output, seed=args.seed, epochs=args.epochs)
    print(json.dumps({"model_version": report["model_version"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
