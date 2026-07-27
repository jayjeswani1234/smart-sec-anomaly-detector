from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CFG
from .utils import build_command_sequence, set_seed


COMMAND_POOL = {
    "user": ["login", "read_file", "search", "download", "logout"],
    "service_account": ["auth", "api_call", "read_config", "write_log", "logout"],
    "edge_device": ["connect", "heartbeat", "telemetry_push", "read_sensor", "logout"],
}

NORMAL_RESOURCES = {
    "user": ["mailbox", "fileshare", "ticketing_portal", "vpn_gateway", "source_repo"],
    "service_account": ["api_gateway", "database", "source_repo", "ticketing_portal"],
    "edge_device": ["iot_hub", "api_gateway", "vpn_gateway"],
}

NORMAL_GEOS = {
    "user": ["India", "Singapore", "United Kingdom"],
    "service_account": ["India", "United States"],
    "edge_device": ["India", "Germany"],
}

AUTH_PREF = {
    "user": ["password", "token", "biometric"],
    "service_account": ["token", "certificate"],
    "edge_device": ["certificate", "token"],
}


@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str
    home_geo: str
    home_hour: int
    usual_resources: List[str]
    auth_method: str
    base_duration: float
    device_fingerprint: str
    access_rate: float


def _random_mac(rng: np.random.Generator) -> str:
    return ":".join(f"{x:02x}" for x in rng.integers(0, 256, size=6))


def _make_profile(rng: np.random.Generator, entity_id: str, entity_type: str) -> EntityProfile:
    home_geo = rng.choice(NORMAL_GEOS[entity_type])
    home_hour = int(rng.normal(10 if entity_type == "user" else 2, 2.5)) % 24
    usual_resources = list(rng.choice(NORMAL_RESOURCES[entity_type], size=min(3, len(NORMAL_RESOURCES[entity_type])), replace=False))
    auth_method = rng.choice(AUTH_PREF[entity_type])
    base_duration = float(max(1.5, rng.normal(12 if entity_type == "user" else 8, 3)))
    device_fingerprint = f"{rng.choice(['win', 'linux', 'ios', 'android'])}-{rng.integers(10,99)}-{_random_mac(rng)}"
    access_rate = float(max(1, rng.normal(6 if entity_type == "user" else 9, 2)))
    return EntityProfile(
        entity_id=entity_id,
        entity_type=entity_type,
        home_geo=home_geo,
        home_hour=home_hour,
        usual_resources=usual_resources,
        auth_method=auth_method,
        base_duration=base_duration,
        device_fingerprint=device_fingerprint,
        access_rate=access_rate,
    )


def _sample_entity_profiles(rng: np.random.Generator, n_entities: int) -> Dict[str, EntityProfile]:
    profiles = {}
    for i in range(n_entities):
        entity_type = rng.choice(CFG.entity_types, p=[0.62, 0.18, 0.20])
        entity_id = f"{entity_type[:1]}_{i:05d}"
        profiles[entity_id] = _make_profile(rng, entity_id, entity_type)
    return profiles


def _make_timestamp(rng: np.random.Generator, base_time: pd.Timestamp, hour_hint: int, drift_days: int = 60) -> pd.Timestamp:
    day_offset = int(rng.integers(0, drift_days))
    minute_offset = int(rng.integers(0, 60))
    hour = int((hour_hint + rng.normal(0, 2)) % 24)
    return base_time + pd.Timedelta(days=day_offset, hours=hour, minutes=minute_offset)


def _make_command_sequence(entity_type: str, anomaly_type: str | None, rng: np.random.Generator) -> str:
    normal_commands = COMMAND_POOL[entity_type]
    if anomaly_type in {"lateral_movement", "credential_stuffing"}:
        commands = ["login", "auth", "list_resources", "escalate", "read_file", "download"]
    elif anomaly_type == "brute_force":
        commands = ["login", "login", "login", "login", "logout"]
    elif anomaly_type == "low_and_slow":
        commands = ["login", "read_file", "search", "read_file", "search", "logout"]
    elif anomaly_type == "device_spoofing":
        commands = ["connect", "heartbeat", "telemetry_push", "logout"]
    else:
        length = int(rng.integers(3, 7))
        commands = list(rng.choice(normal_commands, size=length, replace=True))
    return build_command_sequence(commands)


def _geo_ip_for_geo(geo: str, rng: np.random.Generator) -> str:
    # Simple deterministic IP bands for demo purposes
    bands = {
        "India": "103.",
        "Singapore": "45.",
        "Germany": "88.",
        "United States": "34.",
        "UAE": "5.",
        "United Kingdom": "51.",
        "Japan": "43.",
    }
    prefix = bands.get(geo, "10.")
    return prefix + ".".join(str(int(x)) for x in rng.integers(0, 255, size=3))


def generate_synthetic_logs(
    n_entities: int = CFG.n_entities,
    n_events: int = CFG.n_events,
    anomaly_rate: float = CFG.anomaly_rate,
    seed: int = CFG.seed,
) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    profiles = _sample_entity_profiles(rng, n_entities)
    entity_ids = list(profiles.keys())
    base_time = pd.Timestamp("2026-01-01 00:00:00")

    rows = []
    anomaly_count = max(1, int(n_events * anomaly_rate))

    # Normal events
    for _ in range(n_events - anomaly_count):
        profile = profiles[rng.choice(entity_ids)]
        timestamp = _make_timestamp(rng, base_time, profile.home_hour)
        source_geo = profile.home_geo if rng.random() > 0.08 else rng.choice(CFG.geo_locations)
        resource = rng.choice(profile.usual_resources if rng.random() > 0.15 else CFG.resources)
        auth_method = profile.auth_method if rng.random() > 0.12 else rng.choice(CFG.auth_methods)
        duration = max(0.5, rng.normal(profile.base_duration, 2.0))
        device_fp = profile.device_fingerprint if rng.random() > 0.95 else profile.device_fingerprint
        rows.append(
            {
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": timestamp,
                "source_ip": _geo_ip_for_geo(source_geo, rng),
                "geo_location": source_geo,
                "resource_accessed": resource,
                "auth_method": auth_method,
                "session_duration": round(float(duration), 2),
                "command_sequence": _make_command_sequence(profile.entity_type, None, rng),
                "device_fingerprint": device_fp,
                "label": "normal",
                "anomaly_type": "none",
            }
        )

    # Inject anomalies
    anomaly_types = list(CFG.attack_weights.keys())
    weights = np.array([CFG.attack_weights[a] for a in anomaly_types], dtype=float)
    weights = weights / weights.sum()

    for _ in range(anomaly_count):
        anomaly_type = rng.choice(anomaly_types, p=weights)
        profile = profiles[rng.choice(entity_ids)]

        geo = profile.home_geo
        auth_method = profile.auth_method
        resource = rng.choice(profile.usual_resources)
        duration = max(0.5, rng.normal(profile.base_duration, 2.0))
        device_fp = profile.device_fingerprint
        timestamp = _make_timestamp(rng, base_time, profile.home_hour)

        if anomaly_type == "brute_force":
            auth_method = "password"
            duration = abs(rng.normal(1.0, 0.4))
            resource = "vpn_gateway"
        elif anomaly_type == "impossible_travel":
            geo_choices = [g for g in CFG.geo_locations if g != profile.home_geo]
            geo = rng.choice(geo_choices)
            timestamp = _make_timestamp(rng, base_time + pd.Timedelta(hours=1), (profile.home_hour + 1) % 24)
        elif anomaly_type == "credential_stuffing":
            auth_method = "password"
            resource = "mailbox"
            duration = abs(rng.normal(2.0, 1.0))
        elif anomaly_type == "lateral_movement":
            resource = rng.choice(["admin_panel", "database", "source_repo"])
            duration = abs(rng.normal(profile.base_duration * 1.8, 2.0))
        elif anomaly_type == "device_spoofing":
            device_fp = f"{rng.choice(['win', 'linux', 'mac'])}-{rng.integers(100,999)}-{_random_mac(rng)}"
            auth_method = rng.choice(CFG.auth_methods)
        elif anomaly_type == "low_and_slow":
            duration = abs(rng.normal(profile.base_duration * 3, 1.2))
            resource = rng.choice(profile.usual_resources)
            timestamp = _make_timestamp(rng, base_time, (profile.home_hour + 6) % 24)
        elif anomaly_type == "insider_drift":
            resource = rng.choice(["admin_panel", "database", "payment_service"])
            auth_method = rng.choice(CFG.auth_methods)
            timestamp = _make_timestamp(rng, base_time, (profile.home_hour + 2) % 24)

        rows.append(
            {
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": timestamp,
                "source_ip": _geo_ip_for_geo(geo, rng),
                "geo_location": geo,
                "resource_accessed": resource,
                "auth_method": auth_method,
                "session_duration": round(float(duration), 2),
                "command_sequence": _make_command_sequence(profile.entity_type, anomaly_type, rng),
                "device_fingerprint": device_fp,
                "label": "anomaly",
                "anomaly_type": anomaly_type,
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = [f"evt_{i:06d}" for i in range(len(df))]
    return df


def build_entity_profiles(df: pd.DataFrame) -> pd.DataFrame:
    normal = df[df["label"] == "normal"].copy()
    profile = (
        normal.groupby(["entity_id", "entity_type"])
        .agg(
            home_geo=("geo_location", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            usual_auth=("auth_method", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            avg_duration=("session_duration", "mean"),
            std_duration=("session_duration", "std"),
            primary_resource=("resource_accessed", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            primary_device=("device_fingerprint", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        )
        .reset_index()
    )
    profile["std_duration"] = profile["std_duration"].fillna(profile["avg_duration"] * 0.25 + 1)
    return profile
