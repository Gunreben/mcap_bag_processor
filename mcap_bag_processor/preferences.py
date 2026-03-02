"""
Persistent user preferences stored as JSON.

Location: ~/.config/mcap_bag_processor/preferences.json
"""

import json
from pathlib import Path
from typing import Any, Dict

PREFS_DIR = Path.home() / ".config" / "mcap_bag_processor"
PREFS_FILE = PREFS_DIR / "preferences.json"

DEFAULTS: Dict[str, Any] = {
    "urdf_path": "/home/gunreben/ros2_ws/src/vario700_sensorrig/urdf/vario700_sensorrig_msa.urdf",
    "calibration_dir": "/home/gunreben/ros2_ws/src/tractor_multi_cam_publisher/calibration",
    "last_input_dir": "",
    "last_output_dir": "",
}


def load_preferences() -> Dict[str, Any]:
    """Load preferences from disk, seeding defaults on first launch."""
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                stored = json.load(f)
            merged = {**DEFAULTS, **stored}
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    save_preferences(DEFAULTS)
    return dict(DEFAULTS)


def save_preferences(prefs: Dict[str, Any]) -> None:
    """Write preferences to disk."""
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)
