"""Intermediate result serialization for crash recovery."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _serialize_value(value: Any) -> Any:
    """Convert non-JSON-serializable types to JSON-compatible types."""
    if isinstance(value, (int, str, float, bool, type(None))):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().tolist()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    # Fallback: convert to string
    return str(value)


def save_stage_result(
    stage_name: str, data: dict[str, Any], output_dir: str | Path
) -> None:
    """
    Save stage result to JSON with numpy/tensor conversion.

    Args:
        stage_name: Name of the stage (e.g., "stage_0_vad")
        data: Dictionary to save
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{stage_name}.json"
    serialized = _serialize_value(data)

    with open(result_file, "w") as f:
        json.dump(serialized, f, indent=2)


def load_stage_result(stage_name: str, output_dir: str | Path) -> dict[str, Any] | None:
    """
    Load stage result from JSON.

    Args:
        stage_name: Name of the stage
        output_dir: Output directory

    Returns:
        Data dictionary or None if file doesn't exist
    """
    output_dir = Path(output_dir)
    result_file = output_dir / f"{stage_name}.json"

    if not result_file.exists():
        return None

    with open(result_file) as f:
        return json.load(f)


def get_completed_stages(output_dir: str | Path) -> list[str]:
    """
    Get list of completed stages based on saved intermediate files.

    Args:
        output_dir: Output directory

    Returns:
        List of completed stage names in order
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    completed = []
    for stage_idx in range(4):  # 4 stages total
        stage_name = f"stage_{stage_idx}"
        if (output_dir / f"{stage_name}.json").exists():
            completed.append(stage_name)

    return completed
