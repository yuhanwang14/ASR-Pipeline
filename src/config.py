"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load YAML config, merge env var overrides, validate required fields.

    Args:
        path: Path to config.yaml. If None, uses default "config.yaml"

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If required fields are missing
    """
    if path is None:
        path = "config.yaml"

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Merge environment variable overrides
    # ASR_HF_TOKEN -> diarization.hf_token
    hf_token = os.getenv("ASR_HF_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token and "diarization" in config:
        config["diarization"]["hf_token"] = hf_token

    # Validate required fields
    required_sections = ["audio", "vad", "diarization", "asr", "llm", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config
