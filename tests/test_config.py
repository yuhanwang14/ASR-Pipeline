"""Tests for config loading."""

import os
from pathlib import Path

import pytest

from src.config import load_config


def test_load_config_from_file(tmp_path):
    """Test loading config from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
audio:
  target_sample_rate: 16000
vad:
  model: silero_vad
diarization:
  model: pyannote
asr:
  model: qwen
llm:
  model_id: qwen
output:
  formats: [json]
"""
    )

    config = load_config(str(config_file))
    assert config["audio"]["target_sample_rate"] == 16000
    assert config["vad"]["model"] == "silero_vad"


def test_load_config_default_path(tmp_path):
    """Test loading config from default path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
audio:
  target_sample_rate: 16000
vad:
  model: silero_vad
diarization:
  model: pyannote
asr:
  model: qwen
llm:
  model_id: qwen
output:
  formats: [json]
"""
    )

    # Change to tmp_path directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_config()
        assert config["audio"]["target_sample_rate"] == 16000
    finally:
        os.chdir(original_cwd)


def test_load_config_missing_file():
    """Test error when config file not found."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_load_config_missing_section(tmp_path):
    """Test error when required section is missing."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
audio:
  target_sample_rate: 16000
vad:
  model: silero_vad
"""
    )

    with pytest.raises(ValueError, match="Missing required config section"):
        load_config(str(config_file))


def test_load_config_env_override(tmp_path, monkeypatch):
    """Test HF_TOKEN environment variable override."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
audio:
  target_sample_rate: 16000
vad:
  model: silero_vad
diarization:
  model: pyannote
  hf_token: null
asr:
  model: qwen
llm:
  model_id: qwen
output:
  formats: [json]
"""
    )

    monkeypatch.setenv("ASR_HF_TOKEN", "test-token-123")
    config = load_config(str(config_file))
    assert config["diarization"]["hf_token"] == "test-token-123"


def test_load_config_fallback_hf_token(tmp_path, monkeypatch):
    """Test HF_TOKEN fallback when ASR_HF_TOKEN not set."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
audio:
  target_sample_rate: 16000
vad:
  model: silero_vad
diarization:
  model: pyannote
  hf_token: null
asr:
  model: qwen
llm:
  model_id: qwen
output:
  formats: [json]
"""
    )

    monkeypatch.setenv("HF_TOKEN", "fallback-token")
    monkeypatch.delenv("ASR_HF_TOKEN", raising=False)
    config = load_config(str(config_file))
    assert config["diarization"]["hf_token"] == "fallback-token"
