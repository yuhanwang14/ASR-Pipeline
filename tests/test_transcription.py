"""Tests for transcription module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.transcription import (
    TransformersBackend,
    VLLMBackend,
    _create_backend,
    run_transcription,
)


class MockASRBackend:
    """Mock ASR backend that returns predefined text."""

    def __init__(self, text: str = "Hello world") -> None:
        self.text = text
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        return self.text

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture
def sample_waveform():
    """1-second mono waveform at 16kHz."""
    return torch.randn(1, 16000)


@pytest.fixture
def sample_segments():
    """Two sample segments."""
    return [
        {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def asr_config(sample_config):
    """Config dict from conftest with ASR section."""
    return sample_config


def _mock_save_temp_wav(waveform, sample_rate):
    """Create a real temp file without requiring torchaudio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)
    temp_path.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal fake WAV
    return temp_path


class TestRunTranscription:
    """Test run_transcription function."""

    @patch("src.audio_preprocessing.save_temp_wav", side_effect=_mock_save_temp_wav)
    def test_calls_load_and_unload(self, mock_save, sample_waveform, asr_config):
        """Test that run_transcription calls load() and unload() on backend."""
        backend = MockASRBackend()
        segments = [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]

        run_transcription(sample_waveform, 16000, segments, asr_config, backend=backend)

        assert backend.loaded is True
        assert backend.unloaded is True

    @patch("src.audio_preprocessing.save_temp_wav", side_effect=_mock_save_temp_wav)
    def test_returns_correct_segment_structure(
        self, mock_save, sample_waveform, sample_segments, asr_config
    ):
        """Test that output segments have correct keys and values."""
        backend = MockASRBackend(text="Test transcript")
        results = run_transcription(
            sample_waveform, 16000, sample_segments, asr_config, backend=backend
        )

        assert len(results) == 2
        for i, result in enumerate(results):
            assert "start" in result
            assert "end" in result
            assert "speaker" in result
            assert "text" in result
            assert result["start"] == sample_segments[i]["start"]
            assert result["end"] == sample_segments[i]["end"]
            assert result["speaker"] == sample_segments[i]["speaker"]
            assert result["text"] == "Test transcript"

    @patch("src.audio_preprocessing.save_temp_wav", side_effect=_mock_save_temp_wav)
    def test_handles_multiple_segments(self, mock_save, sample_waveform, asr_config):
        """Test transcription of multiple segments."""
        segments = [
            {"start": 0.0, "end": 0.2, "speaker": "A"},
            {"start": 0.2, "end": 0.4, "speaker": "B"},
            {"start": 0.4, "end": 0.6, "speaker": "A"},
        ]
        backend = MockASRBackend(text="segment text")

        results = run_transcription(sample_waveform, 16000, segments, asr_config, backend=backend)

        assert len(results) == 3
        assert results[0]["speaker"] == "A"
        assert results[1]["speaker"] == "B"
        assert results[2]["speaker"] == "A"
        assert all(r["text"] == "segment text" for r in results)

    def test_cleans_up_temp_files(self, sample_waveform, asr_config):
        """Test that temporary WAV files are cleaned up after transcription."""
        created_paths: list[Path] = []

        def tracking_save(waveform, sample_rate):
            path = _mock_save_temp_wav(waveform, sample_rate)
            created_paths.append(path)
            return path

        segments = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
        ]
        backend = MockASRBackend()

        with patch("src.audio_preprocessing.save_temp_wav", side_effect=tracking_save):
            run_transcription(sample_waveform, 16000, segments, asr_config, backend=backend)

        # All temp files should have been cleaned up
        assert len(created_paths) == 2
        for path in created_paths:
            assert not path.exists(), f"Temp file not cleaned up: {path}"

    @patch("src.audio_preprocessing.save_temp_wav", side_effect=_mock_save_temp_wav)
    def test_unload_called_on_transcribe_error(self, mock_save, sample_waveform, asr_config):
        """Test that unload() is called even if transcribe() raises."""
        backend = MockASRBackend()
        backend.transcribe = MagicMock(side_effect=RuntimeError("ASR failed"))

        segments = [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]

        with pytest.raises(RuntimeError, match="ASR failed"):
            run_transcription(sample_waveform, 16000, segments, asr_config, backend=backend)

        assert backend.unloaded is True

    @patch("src.audio_preprocessing.save_temp_wav", side_effect=_mock_save_temp_wav)
    def test_segment_without_speaker_defaults_to_unknown(
        self, mock_save, sample_waveform, asr_config
    ):
        """Test that segments without a speaker key default to UNKNOWN."""
        backend = MockASRBackend(text="some text")
        segments = [{"start": 0.0, "end": 0.5}]

        results = run_transcription(sample_waveform, 16000, segments, asr_config, backend=backend)

        assert results[0]["speaker"] == "UNKNOWN"


class TestBackendConfig:
    """Test backend instantiation and config storage."""

    def test_vllm_backend_stores_config(self, asr_config):
        """Test that VLLMBackend stores config without loading."""
        backend = VLLMBackend(asr_config)
        assert backend.config is asr_config
        assert backend.model is None

    def test_transformers_backend_stores_config(self, asr_config):
        """Test that TransformersBackend stores config without loading."""
        backend = TransformersBackend(asr_config)
        assert backend.config is asr_config
        assert backend.model is None


class TestBackendSelection:
    """Test backend selection based on config."""

    def test_select_vllm_backend(self, asr_config):
        """Test that 'vllm' config selects VLLMBackend."""
        asr_config["asr"]["backend"] = "vllm"
        backend = _create_backend(asr_config)
        assert isinstance(backend, VLLMBackend)

    def test_select_transformers_backend(self, asr_config):
        """Test that 'transformers' config selects TransformersBackend."""
        asr_config["asr"]["backend"] = "transformers"
        backend = _create_backend(asr_config)
        assert isinstance(backend, TransformersBackend)

    def test_unknown_backend_raises(self, asr_config):
        """Test that unknown backend raises ValueError."""
        asr_config["asr"]["backend"] = "whisper"
        with pytest.raises(ValueError, match="Unknown ASR backend"):
            _create_backend(asr_config)

    def test_default_backend_is_vllm(self, asr_config):
        """Test that missing backend key defaults to vllm."""
        del asr_config["asr"]["backend"]
        backend = _create_backend(asr_config)
        assert isinstance(backend, VLLMBackend)
