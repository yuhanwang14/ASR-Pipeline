"""Shared pytest fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_output_dir() -> Path:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> dict:
    """Minimal sample config for testing."""
    return {
        "audio": {
            "target_sample_rate": 16000,
            "target_channels": 1,
        },
        "vad": {
            "model": "silero_vad",
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "merge_gap_seconds": 0.5,
        },
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "hf_token": None,
            "num_speakers": None,
            "speaker_profiles_dir": "speaker_profiles/",
            "match_threshold": 0.75,
        },
        "asr": {
            "model": "Qwen/Qwen3-ASR-1.7B",
            "backend": "vllm",
            "dtype": "bfloat16",
            "max_new_tokens": 4096,
            "language": None,
            "max_segment_duration": 300,
            "gpu_memory_utilization": 0.7,
            "flash_attention": True,
        },
        "llm": {
            "model_path": None,
            "model_id": "Qwen/Qwen3.5-9B",
            "gguf_url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
            "backend": "llama-cpp",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "tasks": {
                "speaker_correction": True,
                "text_correction": True,
            },
        },
        "output": {
            "formats": ["json", "srt", "txt"],
            "output_dir": "output/",
            "save_intermediate": True,
        },
    }
