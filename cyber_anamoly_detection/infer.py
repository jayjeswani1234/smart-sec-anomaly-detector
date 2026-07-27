from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.explain import format_explanation
from src.model import load_bundle, score_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Score access logs and rank alerts.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output", type=str, default="scored_events.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bundle = load_bundle()
    scored = score_events(df, bundle)
    scored["explanation"] = scored.apply(format_explanation, axis=1)
    scored.to_csv(args.output, index=False)
    print(f"Saved scored alerts to {args.output}")


if __name__ == "__main__":
    main()
