"""Tests for VAD module using mocks."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.vad import run_vad


def _mock_load_silero(mock_load_fn, mock_get_timestamps_fn):
    """Create a mock _load_silero that returns the given mock functions."""
    return lambda: (mock_load_fn, mock_get_timestamps_fn)


class TestRunVad:
    """Test run_vad function."""

    def _make_config(self, merge_gap_seconds=0.5):
        """Create a minimal config dict for VAD tests."""
        return {
            "vad": {
                "model": "silero_vad",
                "merge_gap_seconds": merge_gap_seconds,
            }
        }

    def _patch_silero(self, speech_timestamps):
        """Create a patch for _load_silero that returns given timestamps.

        Returns (patcher, mock_load_vad, mock_get_timestamps) tuple.
        Use as a context manager or call .start()/.stop().
        """
        mock_load_vad = MagicMock(return_value=MagicMock())
        mock_get_timestamps = MagicMock(return_value=speech_timestamps)
        patcher = patch(
            "src.vad._load_silero",
            return_value=(mock_load_vad, mock_get_timestamps),
        )
        return patcher, mock_load_vad, mock_get_timestamps

    def test_run_vad_returns_correct_structure(self):
        """Test that run_vad returns (clean_waveform, timestamp_map, segments)."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 0.5, "end": 1.5},
                {"start": 3.0, "end": 4.0},
            ]
        )

        with patcher:
            waveform = torch.randn(1, 16000 * 5)  # 5 seconds
            config = self._make_config()
            clean_waveform, timestamp_map, speech_segments = run_vad(waveform, 16000, config)

        assert isinstance(clean_waveform, torch.Tensor)
        assert clean_waveform.dim() == 2
        assert clean_waveform.shape[0] == 1

        assert isinstance(timestamp_map, list)
        assert all(isinstance(entry, dict) for entry in timestamp_map)

        assert isinstance(speech_segments, list)
        assert all(isinstance(seg, dict) for seg in speech_segments)

    def test_run_vad_clean_waveform_is_concatenation(self):
        """Test that clean_waveform is the concatenation of speech segments."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 0.0, "end": 1.0},
                {"start": 2.0, "end": 3.0},
            ]
        )

        # Create waveform with known values
        sample_rate = 16000
        waveform = torch.zeros(1, sample_rate * 4)  # 4 seconds
        # Fill 0-1s with 1.0
        waveform[0, :sample_rate] = 1.0
        # Fill 2-3s with 2.0
        waveform[0, 2 * sample_rate : 3 * sample_rate] = 2.0

        config = self._make_config(merge_gap_seconds=0.1)  # Won't merge (gap=1.0s)
        with patcher:
            clean_waveform, timestamp_map, speech_segments = run_vad(waveform, sample_rate, config)

        # Clean waveform should be 2 seconds of speech
        assert clean_waveform.shape[1] == 2 * sample_rate

        # First second should be all 1.0
        assert torch.all(clean_waveform[0, :sample_rate] == 1.0)
        # Second second should be all 2.0
        assert torch.all(clean_waveform[0, sample_rate:] == 2.0)

    def test_run_vad_timestamp_map_structure(self):
        """Test that timestamp_map has correct keys and values."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 1.0, "end": 2.0},
                {"start": 5.0, "end": 6.0},
            ]
        )

        waveform = torch.randn(1, 16000 * 8)  # 8 seconds
        config = self._make_config(merge_gap_seconds=0.1)

        with patcher:
            _, timestamp_map, _ = run_vad(waveform, 16000, config)

        assert len(timestamp_map) == 2

        # First entry
        assert timestamp_map[0]["original_start"] == pytest.approx(1.0)
        assert timestamp_map[0]["original_end"] == pytest.approx(2.0)
        assert timestamp_map[0]["clean_start"] == pytest.approx(0.0)
        assert timestamp_map[0]["clean_end"] == pytest.approx(1.0)

        # Second entry: clean starts where first ended
        assert timestamp_map[1]["original_start"] == pytest.approx(5.0)
        assert timestamp_map[1]["original_end"] == pytest.approx(6.0)
        assert timestamp_map[1]["clean_start"] == pytest.approx(1.0)
        assert timestamp_map[1]["clean_end"] == pytest.approx(2.0)

    def test_run_vad_merge_close_segments(self):
        """Test that close segments are merged according to config."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 0.0, "end": 1.0},
                {"start": 1.2, "end": 2.0},  # Gap 0.2 < 0.5 -> merged
                {"start": 5.0, "end": 6.0},  # Gap 3.0 > 0.5 -> separate
            ]
        )

        waveform = torch.randn(1, 16000 * 8)
        config = self._make_config(merge_gap_seconds=0.5)

        with patcher:
            _, timestamp_map, speech_segments = run_vad(waveform, 16000, config)

        # After merge: [0.0-2.0] and [5.0-6.0]
        assert len(speech_segments) == 2
        assert speech_segments[0]["start"] == pytest.approx(0.0)
        assert speech_segments[0]["end"] == pytest.approx(2.0)
        assert speech_segments[1]["start"] == pytest.approx(5.0)
        assert speech_segments[1]["end"] == pytest.approx(6.0)

        # Timestamp map should also have 2 entries
        assert len(timestamp_map) == 2

    def test_run_vad_no_speech(self):
        """Test run_vad when no speech is detected."""
        patcher, _, _ = self._patch_silero([])

        waveform = torch.randn(1, 16000 * 2)
        config = self._make_config()

        with patcher:
            clean_waveform, timestamp_map, speech_segments = run_vad(waveform, 16000, config)

        assert clean_waveform.shape[1] == 0
        assert timestamp_map == []
        assert speech_segments == []

    def test_run_vad_calls_silero_correctly(self):
        """Test that Silero VAD functions are called with correct arguments."""
        mock_model = MagicMock()
        mock_load_vad = MagicMock(return_value=mock_model)
        mock_get_timestamps = MagicMock(return_value=[{"start": 0.0, "end": 1.0}])

        with patch(
            "src.vad._load_silero",
            return_value=(mock_load_vad, mock_get_timestamps),
        ):
            waveform = torch.randn(1, 16000)
            config = self._make_config()
            run_vad(waveform, 16000, config)

        # load_silero_vad called once
        mock_load_vad.assert_called_once()

        # get_speech_timestamps called with squeezed waveform, model, return_seconds=True
        mock_get_timestamps.assert_called_once()
        call_args = mock_get_timestamps.call_args
        passed_wav = call_args[0][0]
        assert passed_wav.dim() == 1  # Squeezed from (1, N) to (N,)
        assert call_args[0][1] is mock_model
        assert call_args[1]["return_seconds"] is True

    def test_run_vad_single_segment(self):
        """Test run_vad with a single speech segment."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 0.0, "end": 2.0},
            ]
        )

        sample_rate = 16000
        waveform = torch.ones(1, sample_rate * 3)
        config = self._make_config()

        with patcher:
            clean_waveform, timestamp_map, speech_segments = run_vad(waveform, sample_rate, config)

        assert len(speech_segments) == 1
        assert len(timestamp_map) == 1
        assert clean_waveform.shape[1] == 2 * sample_rate

        # Clean offset should start at 0 and end at segment duration
        assert timestamp_map[0]["clean_start"] == pytest.approx(0.0)
        assert timestamp_map[0]["clean_end"] == pytest.approx(2.0)

    def test_run_vad_default_merge_gap(self):
        """Test that default merge gap is used when config key is missing."""
        patcher, _, _ = self._patch_silero(
            [
                {"start": 0.0, "end": 1.0},
                {"start": 1.3, "end": 2.0},  # Gap 0.3 < default 0.5 -> merged
            ]
        )

        waveform = torch.randn(1, 16000 * 3)
        config = {}  # No vad section

        with patcher:
            _, _, speech_segments = run_vad(waveform, 16000, config)

        # Should merge with default gap of 0.5
        assert len(speech_segments) == 1
