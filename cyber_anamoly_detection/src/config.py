from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_entities: int = 250
    n_events: int = 8000
    anomaly_rate: float = 0.03
    top_alert_budget: float = 0.01  # top 1% of events
    test_size: float = 0.2
    model_dir: str = "models"
    data_dir: str = "data"

    entity_types: List[str] = field(default_factory=lambda: ["user", "service_account", "edge_device"])
    auth_methods: List[str] = field(default_factory=lambda: ["password", "token", "certificate", "biometric"])
    resources: List[str] = field(default_factory=lambda: [
        "vpn_gateway",
        "mailbox",
        "fileshare",
        "admin_panel",
        "api_gateway",
        "source_repo",
        "iot_hub",
        "database",
        "payment_service",
        "ticketing_portal",
    ])
    geo_locations: List[str] = field(default_factory=lambda: [
        "India",
        "Singapore",
        "Germany",
        "United States",
        "UAE",
        "United Kingdom",
        "Japan",
    ])
    attack_types: List[str] = field(default_factory=lambda: [
        "brute_force",
        "impossible_travel",
        "credential_stuffing",
        "lateral_movement",
        "device_spoofing",
        "low_and_slow",
        "insider_drift",
    ])

    attack_weights: Dict[str, float] = field(default_factory=lambda: {
        "brute_force": 0.18,
        "impossible_travel": 0.15,
        "credential_stuffing": 0.17,
        "lateral_movement": 0.17,
        "device_spoofing": 0.14,
        "low_and_slow": 0.12,
        "insider_drift": 0.07,
    })


CFG = Config()
