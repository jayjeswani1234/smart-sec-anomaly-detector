from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .utils import parse_command_sequence, cyclical_hour_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(int)
    out["is_off_hours"] = ((out["hour"] < 7) | (out["hour"] > 20)).astype(int)
    out["command_len"] = out["command_sequence"].fillna("").map(lambda x: len(parse_command_sequence(x)))
    out["unique_commands"] = out["command_sequence"].fillna("").map(lambda x: len(set(parse_command_sequence(x))))
    out["is_password_auth"] = (out["auth_method"].fillna("") == "password").astype(int)
    out = pd.concat([out, cyclical_hour_features(out["hour"])], axis=1)
    return out


def add_entity_deviation_features(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    merged = out.merge(profiles, on=["entity_id", "entity_type"], how="left")
    out["home_geo_match"] = (merged["geo_location"] == merged["home_geo"]).astype(int)
    out["usual_auth_match"] = (merged["auth_method"] == merged["usual_auth"]).astype(int)
    out["usual_resource_match"] = (merged["resource_accessed"] == merged["primary_resource"]).astype(int)
    out["known_device_match"] = (merged["device_fingerprint"] == merged["primary_device"]).astype(int)

    out["duration_z"] = (merged["session_duration"] - merged["avg_duration"]) / (merged["std_duration"].replace(0, np.nan))
    out["duration_z"] = out["duration_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # A simple concept-drift score proxy: off-hours + geo/resource mismatch + longer duration.
    out["drift_score"] = (
        (1 - out["home_geo_match"]) * 1.5
        + (1 - out["usual_resource_match"]) * 0.8
        + out["is_off_hours"] * 0.7
        + out["duration_z"].clip(lower=0) * 0.25
    )
    return out


def select_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric = [
        "session_duration",
        "hour",
        "dayofweek",
        "is_off_hours",
        "command_len",
        "unique_commands",
        "is_password_auth",
        "home_geo_match",
        "usual_auth_match",
        "usual_resource_match",
        "known_device_match",
        "duration_z",
        "drift_score",
        "hour_sin",
        "hour_cos",
    ]
    categorical = [
        "entity_type",
        "geo_location",
        "auth_method",
        "resource_accessed",
    ]
    return numeric, categorical
