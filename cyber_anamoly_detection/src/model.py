from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CFG
from .features import add_entity_deviation_features, add_time_features, select_feature_columns
from .synthetic_data import build_entity_profiles, generate_synthetic_logs
from .utils import ensure_dir, save_json


@dataclass
class TrainedBundle:
    anomaly_model: Pipeline
    attack_model: Pipeline
    metadata: Dict


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    profiles = build_entity_profiles(df)
    out = add_time_features(df)
    out = add_entity_deviation_features(out, profiles)
    return out


def train_models(df: pd.DataFrame | None = None, seed: int = CFG.seed) -> TrainedBundle:
    if df is None:
        df = generate_synthetic_logs(seed=seed)
    df = prepare_dataset(df)

    numeric_cols, categorical_cols = select_feature_columns(df)
    feature_cols = numeric_cols + categorical_cols

    # Train anomaly detector on normal samples only.
    train_df = df[df["label"] == "normal"].copy()
    X_train = train_df[feature_cols]

    anomaly_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    anomaly_model = Pipeline(
        steps=[
            ("prep", anomaly_preprocessor),
            ("clf", IsolationForest(
                n_estimators=250,
                contamination=CFG.anomaly_rate,
                random_state=seed,
                bootstrap=False,
            )),
        ]
    )
    anomaly_model.fit(X_train)

    # Attack classifier on anomalous samples.
    attack_df = df[df["label"] == "anomaly"].copy()
    attack_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    attack_model = Pipeline(
        steps=[
            ("prep", attack_preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight="balanced_subsample",
                max_depth=14,
            )),
        ]
    )
    attack_model.fit(attack_df[feature_cols], attack_df["anomaly_type"])

    # Evaluate on a mixed sample
    X_all = df[feature_cols]
    anomaly_scores = -anomaly_model.named_steps["clf"].decision_function(anomaly_model.named_steps["prep"].transform(X_all))
    y_true = (df["label"] == "anomaly").astype(int).values
    y_pred = (anomaly_scores >= np.quantile(anomaly_scores, 1 - CFG.top_alert_budget)).astype(int)
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except Exception:
        auc = float("nan")

    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    meta = {
        "seed": seed,
        "n_rows": int(len(df)),
        "anomaly_rate": float(df["label"].eq("anomaly").mean()),
        "roc_auc": float(auc),
        "precision_at_budget": float(pr),
        "recall_at_budget": float(rc),
        "f1_at_budget": float(f1),
        "feature_columns": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    return TrainedBundle(anomaly_model=anomaly_model, attack_model=attack_model, metadata=meta)


def save_bundle(bundle: TrainedBundle, model_dir: str = CFG.model_dir) -> None:
    model_path = ensure_dir(model_dir)
    joblib.dump(bundle.anomaly_model, model_path / "anomaly_detector.joblib")
    joblib.dump(bundle.attack_model, model_path / "attack_classifier.joblib")
    save_json(model_path / "metadata.json", bundle.metadata)


def load_bundle(model_dir: str = CFG.model_dir) -> TrainedBundle:
    model_path = Path(model_dir)
    anomaly_model = joblib.load(model_path / "anomaly_detector.joblib")
    attack_model = joblib.load(model_path / "attack_classifier.joblib")
    import json
    metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    return TrainedBundle(anomaly_model=anomaly_model, attack_model=attack_model, metadata=metadata)


def score_events(df: pd.DataFrame, bundle: TrainedBundle) -> pd.DataFrame:
    df = prepare_dataset(df)
    feature_cols = bundle.metadata["feature_columns"]
    X = df[feature_cols]

    prep = bundle.anomaly_model.named_steps["prep"]
    clf = bundle.anomaly_model.named_steps["clf"]
    transformed = prep.transform(X)
    anomaly_score = -clf.decision_function(transformed)
    threshold = np.quantile(anomaly_score, 1 - CFG.top_alert_budget)
    predicted_anomaly = anomaly_score >= threshold

    attack_pred = bundle.attack_model.predict(X)
    attack_prob = None
    if hasattr(bundle.attack_model.named_steps["clf"], "predict_proba"):
        attack_prob = bundle.attack_model.predict_proba(X).max(axis=1)
    else:
        attack_prob = np.ones(len(df))

    out = df.copy()
    out["anomaly_score"] = anomaly_score
    out["alert_threshold"] = threshold
    out["predicted_anomaly"] = predicted_anomaly
    out["predicted_attack_type"] = attack_pred
    out["attack_confidence"] = attack_prob
    out["alert_risk"] = np.clip(
        (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9),
        0,
        1,
    ) * 0.7 + np.clip(attack_prob, 0, 1) * 0.3
    return out.sort_values("alert_risk", ascending=False).reset_index(drop=True)


def train_and_save(seed: int = CFG.seed, model_dir: str = CFG.model_dir) -> TrainedBundle:
    bundle = train_models(seed=seed)
    save_bundle(bundle, model_dir=model_dir)
    return bundle
