from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CFG
from src.model import train_and_save
from src.synthetic_data import generate_synthetic_logs
from src.utils import ensure_dir


def main() -> None:
    ensure_dir(CFG.data_dir)
    ensure_dir(CFG.model_dir)

    df = generate_synthetic_logs(
        n_entities=CFG.n_entities,
        n_events=CFG.n_events,
        anomaly_rate=CFG.anomaly_rate,
        seed=CFG.seed,
    )
    df.to_csv(Path(CFG.data_dir) / "synthetic_access_logs.csv", index=False)

    bundle = train_and_save(seed=CFG.seed, model_dir=CFG.model_dir)

    print("Training complete.")
    print(bundle.metadata)
    print(f"Saved dataset to {Path(CFG.data_dir) / 'synthetic_access_logs.csv'}")
    print(f"Saved models to {CFG.model_dir}/")


if __name__ == "__main__":
    main()
