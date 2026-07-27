from __future__ import annotations

from typing import List

import pandas as pd


def explain_row(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    if row.get("is_off_hours", 0) == 1:
        reasons.append("Access occurred outside usual working hours.")

    if row.get("home_geo_match", 1) == 0:
        reasons.append(f"Access came from unusual geo location: {row.get('geo_location')}.")

    if row.get("usual_auth_match", 1) == 0:
        reasons.append(f"Authentication method is unusual: {row.get('auth_method')}.")

    if row.get("usual_resource_match", 1) == 0:
        reasons.append(f"Resource accessed is unusual: {row.get('resource_accessed')}.")

    if row.get("known_device_match", 1) == 0:
        reasons.append("Device fingerprint does not match historical device profile.")

    if row.get("duration_z", 0) > 1.5:
        reasons.append("Session duration is significantly higher than the entity baseline.")

    if row.get("command_len", 0) >= 5 and row.get("unique_commands", 0) <= 3:
        reasons.append("Command sequence looks repetitive and suspicious.")

    if row.get("drift_score", 0) > 1.5:
        reasons.append("Behavior indicates possible concept drift or new usage pattern.")

    if not reasons:
        reasons.append("Event deviates from the learned baseline across multiple weak signals.")

    return reasons[:4]


def format_explanation(row: pd.Series) -> str:
    reasons = explain_row(row)
    return " | ".join(reasons)
