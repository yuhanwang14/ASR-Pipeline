"""Tests for the diarization module."""

from unittest.mock import MagicMock, patch

import pytest

from src.diarization import DiarizationBackend, PyAnnoteBackend, run_diarization


class MockDiarizationBackend:
    """Mock backend that tracks call order and returns predefined segments."""

    def __init__(self, segments: list[dict] | None = None) -> None:
        self.segments = segments or [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]
        self.call_order: list[str] = []

    def load(self) -> None:
        self.call_order.append("load")

    def run(self, waveform, sample_rate: int) -> list[dict]:
        self.call_order.append("run")
        return self.segments

    def unload(self) -> None:
        self.call_order.append("unload")


class TestDiarizationBackendProtocol:
    """Test that MockDiarizationBackend satisfies the Protocol."""

    def test_mock_is_diarization_backend(self):
        """MockDiarizationBackend satisfies DiarizationBackend protocol."""
        backend = MockDiarizationBackend()
        assert isinstance(backend, DiarizationBackend)


class TestRunDiarization:
    """Test run_diarization orchestration."""

    def test_calls_load_run_unload_in_order(self):
        """Verify load, run, unload are called in correct sequence."""
        backend = MockDiarizationBackend()
        waveform = MagicMock()
        config = {"diarization": {"model": "test"}}

        run_diarization(waveform, 16000, config, backend=backend)

        assert backend.call_order == ["load", "run", "unload"]

    def test_returns_correct_segments(self):
        """Verify returned segments match backend output."""
        expected = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.0, "end": 4.0, "speaker": "B"},
        ]
        backend = MockDiarizationBackend(segments=expected)
        waveform = MagicMock()
        config = {"diarization": {}}

        result = run_diarization(waveform, 16000, config, backend=backend)

        assert result == expected

    def test_segment_structure(self):
        """Each segment has start, end, and speaker keys."""
        backend = MockDiarizationBackend()
        waveform = MagicMock()
        config = {"diarization": {}}

        result = run_diarization(waveform, 16000, config, backend=backend)

        for seg in result:
            assert "start" in seg
            assert "end" in seg
            assert "speaker" in seg
            assert isinstance(seg["start"], float)
            assert isinstance(seg["end"], float)
            assert isinstance(seg["speaker"], str)

    def test_unload_called_on_run_failure(self):
        """Unload is called even if run() raises an exception."""
        backend = MockDiarizationBackend()

        def failing_run(waveform, sample_rate):
            backend.call_order.append("run")
            raise RuntimeError("inference failed")

        backend.run = failing_run
        waveform = MagicMock()
        config = {"diarization": {}}

        with pytest.raises(RuntimeError, match="inference failed"):
            run_diarization(waveform, 16000, config, backend=backend)

        assert "unload" in backend.call_order

    def test_creates_default_backend_when_none(self):
        """When no backend is given, PyAnnoteBackend is created and used."""
        waveform = MagicMock()
        config = {
            "diarization": {
                "model": "pyannote/speaker-diarization-3.1",
                "hf_token": "test_token",
            }
        }

        with (
            patch.object(PyAnnoteBackend, "load") as mock_load,
            patch.object(PyAnnoteBackend, "run", return_value=[]) as mock_run,
            patch.object(PyAnnoteBackend, "unload") as mock_unload,
        ):
            result = run_diarization(waveform, 16000, config)

            mock_load.assert_called_once()
            mock_run.assert_called_once_with(waveform, 16000)
            mock_unload.assert_called_once()
            assert result == []


class TestPyAnnoteBackend:
    """Test PyAnnoteBackend configuration (no actual model loading)."""

    def test_stores_config(self):
        """Backend stores config values correctly."""
        config = {
            "diarization": {
                "model": "pyannote/speaker-diarization-3.1",
                "hf_token": "my_token",
                "num_speakers": 3,
            }
        }
        backend = PyAnnoteBackend(config)

        assert backend._model_name == "pyannote/speaker-diarization-3.1"
        assert backend._hf_token == "my_token"
        assert backend._num_speakers == 3
        assert backend._pipeline is None

    def test_default_config_values(self):
        """Backend uses defaults when config keys are missing."""
        config = {"diarization": {}}
        backend = PyAnnoteBackend(config)

        assert backend._model_name == "pyannote/speaker-diarization-3.1"
        assert backend._hf_token is None
        assert backend._num_speakers is None

    def test_empty_config(self):
        """Backend handles completely empty config."""
        backend = PyAnnoteBackend({})

        assert backend._model_name == "pyannote/speaker-diarization-3.1"
        assert backend._hf_token is None
        assert backend._num_speakers is None

    def test_run_without_load_raises(self):
        """Calling run() before load() raises RuntimeError."""
        backend = PyAnnoteBackend({"diarization": {}})
        waveform = MagicMock()

        with pytest.raises(RuntimeError, match="Backend not loaded"):
            backend.run(waveform, 16000)

    def test_satisfies_protocol(self):
        """PyAnnoteBackend satisfies DiarizationBackend protocol."""
        assert isinstance(PyAnnoteBackend({}), DiarizationBackend)

    def test_unload_when_not_loaded(self):
        """Unload is safe to call even if load() was never called."""
        backend = PyAnnoteBackend({"diarization": {}})
        # Should not raise
        backend.unload()
        assert backend._pipeline is None
