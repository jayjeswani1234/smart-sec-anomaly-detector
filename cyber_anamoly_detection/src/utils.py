from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_command_sequence(seq: str) -> List[str]:
    if not isinstance(seq, str) or not seq.strip():
        return []
    return [x.strip() for x in seq.split(">") if x.strip()]


def build_command_sequence(commands: Iterable[str]) -> str:
    return " > ".join(commands)


def cyclical_hour_features(hour: pd.Series) -> pd.DataFrame:
    radians = 2 * np.pi * hour / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(radians),
            "hour_cos": np.cos(radians),
        },
        index=hour.index,
    )
