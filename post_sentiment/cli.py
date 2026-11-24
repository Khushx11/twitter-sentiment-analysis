import argparse
import json
from pathlib import Path
from .artefacts import load_trusted
from .inference import predict
from .workflow import run

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and use a three-class short-post sentiment model.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("csv", type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--seed", default=42, type=int)
    inference = commands.add_parser("predict")
    inference.add_argument("model", type=Path, help="Trusted local joblib file produced by train")
    inference.add_argument("text")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = run(args.csv, args.output, args.seed)
        else:
            result = predict(load_trusted(args.model), args.text)
    except (ValueError, OSError) as error:
        parser.exit(1, f"post-sentiment: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
