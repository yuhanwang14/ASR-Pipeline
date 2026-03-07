"""Tests for the speaker registry module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.speaker_registry import (
    _cosine_similarity,
    delete_speaker,
    enroll_speaker,
    list_enrolled_speakers,
    match_speakers,
)


@pytest.fixture
def registry_config(tmp_path: Path) -> dict:
    """Config with a temporary speaker_profiles_dir."""
    return {
        "diarization": {
            "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "hf_token": None,
            "speaker_profiles_dir": str(tmp_path / "profiles"),
            "match_threshold": 0.75,
        }
    }


@pytest.fixture
def profiles_dir(registry_config: dict) -> Path:
    """Create and return the profiles directory."""
    d = Path(registry_config["diarization"]["speaker_profiles_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestEnrollSpeaker:
    """Test enroll_speaker function."""

    def test_saves_npy_file(self, registry_config: dict, tmp_path: Path):
        """Enrollment saves a .npy file with the averaged embedding."""
        fake_embedding = np.random.randn(1, 256).astype(np.float32)

        mock_model = MagicMock()
        mock_inference = MagicMock()
        mock_inference.return_value = fake_embedding

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model
        mock_pyannote_audio.Inference.return_value = mock_inference

        audio_file = tmp_path / "sample.wav"
        audio_file.touch()

        with (
            patch.dict(
                sys.modules,
                {"pyannote": MagicMock(), "pyannote.audio": mock_pyannote_audio},
            ),
            patch("src.gpu_utils.unload_model"),
        ):
            enroll_speaker("Alice", [str(audio_file)], registry_config)

        profiles_dir = Path(registry_config["diarization"]["speaker_profiles_dir"])
        saved_path = profiles_dir / "Alice.npy"
        assert saved_path.exists()

        loaded = np.load(saved_path)
        np.testing.assert_array_almost_equal(loaded, fake_embedding)

    def test_averages_multiple_audio_files(self, registry_config: dict, tmp_path: Path):
        """Enrollment averages embeddings from multiple audio files."""
        emb1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        emb2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        expected_avg = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)

        mock_inference = MagicMock()
        mock_inference.side_effect = [emb1, emb2]

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Model.from_pretrained.return_value = MagicMock()
        mock_pyannote_audio.Inference.return_value = mock_inference

        audio1 = tmp_path / "a1.wav"
        audio2 = tmp_path / "a2.wav"
        audio1.touch()
        audio2.touch()

        with (
            patch.dict(
                sys.modules,
                {"pyannote": MagicMock(), "pyannote.audio": mock_pyannote_audio},
            ),
            patch("src.gpu_utils.unload_model"),
        ):
            enroll_speaker("Bob", [str(audio1), str(audio2)], registry_config)

        profiles_dir = Path(registry_config["diarization"]["speaker_profiles_dir"])
        loaded = np.load(profiles_dir / "Bob.npy")
        np.testing.assert_array_almost_equal(loaded, expected_avg)

    def test_raises_on_empty_audio_paths(self, registry_config: dict):
        """Enrollment raises ValueError when no audio files provided."""
        with pytest.raises(ValueError, match="At least one audio file"):
            enroll_speaker("Nobody", [], registry_config)

    def test_raises_on_missing_audio_file(self, registry_config: dict):
        """Enrollment raises FileNotFoundError for nonexistent audio."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            enroll_speaker("Ghost", ["/nonexistent/audio.wav"], registry_config)


class TestMatchSpeakers:
    """Test match_speakers function."""

    def test_matches_above_threshold(self, registry_config: dict, profiles_dir: Path):
        """Speaker with embedding close to profile is matched."""
        profile_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.save(profiles_dir / "Alice.npy", profile_emb)

        # Embedding very close to Alice's profile
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "embedding": np.array([0.95, 0.05, 0.0]),
            },
            {
                "start": 2.0,
                "end": 4.0,
                "speaker": "SPEAKER_00",
                "embedding": np.array([0.9, 0.1, 0.0]),
            },
        ]

        result = match_speakers(segments, registry_config)

        assert all(seg["speaker"] == "Alice" for seg in result)

    def test_no_match_below_threshold(self, registry_config: dict, profiles_dir: Path):
        """Speaker with embedding far from all profiles keeps original label."""
        profile_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.save(profiles_dir / "Alice.npy", profile_emb)

        # Embedding orthogonal to Alice's
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "embedding": np.array([0.0, 1.0, 0.0]),
            },
        ]

        result = match_speakers(segments, registry_config)

        assert result[0]["speaker"] == "SPEAKER_00"

    def test_threshold_boundary(self, registry_config: dict, profiles_dir: Path):
        """Test behavior right at the cosine similarity threshold."""
        # Create a profile and a segment embedding whose similarity is exactly at threshold
        profile_emb = np.array([1.0, 0.0], dtype=np.float32)
        np.save(profiles_dir / "Alice.npy", profile_emb)

        # cos(angle) = 0.75 -> angle ~41.4 degrees
        # [0.75, sqrt(1-0.75^2)] = [0.75, 0.6614]
        seg_emb = np.array([0.75, np.sqrt(1 - 0.75**2)])

        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "embedding": seg_emb}]

        result = match_speakers(segments, registry_config)
        # Similarity is exactly 0.75, threshold is 0.75 (>=), should match
        assert result[0]["speaker"] == "Alice"

    def test_just_below_threshold(self, registry_config: dict, profiles_dir: Path):
        """Similarity just below threshold does not match."""
        profile_emb = np.array([1.0, 0.0], dtype=np.float32)
        np.save(profiles_dir / "Alice.npy", profile_emb)

        # cos(angle) slightly below 0.75
        seg_emb = np.array([0.74, np.sqrt(1 - 0.74**2)])

        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "embedding": seg_emb}]

        result = match_speakers(segments, registry_config)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_no_profiles_returns_unchanged(self, registry_config: dict, profiles_dir: Path):
        """With no enrolled profiles, segments are returned unchanged."""
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ]

        result = match_speakers(segments, registry_config)

        assert result == segments

    def test_segments_without_embeddings(self, registry_config: dict, profiles_dir: Path):
        """Segments without embedding field are left unmatched."""
        profile_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.save(profiles_dir / "Alice.npy", profile_emb)

        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ]

        result = match_speakers(segments, registry_config)

        assert result[0]["speaker"] == "SPEAKER_00"

    def test_multiple_speakers_matched(self, registry_config: dict, profiles_dir: Path):
        """Multiple anonymous speakers can each match different profiles."""
        np.save(profiles_dir / "Alice.npy", np.array([1.0, 0.0, 0.0]))
        np.save(profiles_dir / "Bob.npy", np.array([0.0, 1.0, 0.0]))

        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "embedding": np.array([0.98, 0.02, 0.0]),
            },
            {
                "start": 2.0,
                "end": 4.0,
                "speaker": "SPEAKER_01",
                "embedding": np.array([0.02, 0.98, 0.0]),
            },
        ]

        result = match_speakers(segments, registry_config)

        assert result[0]["speaker"] == "Alice"
        assert result[1]["speaker"] == "Bob"

    def test_does_not_mutate_input(self, registry_config: dict, profiles_dir: Path):
        """match_speakers returns new dicts, not modified originals."""
        np.save(profiles_dir / "Alice.npy", np.array([1.0, 0.0]))

        original_seg = {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "embedding": np.array([0.99, 0.01]),
        }
        segments = [original_seg]

        result = match_speakers(segments, registry_config)

        # Original segment should be unchanged
        assert original_seg["speaker"] == "SPEAKER_00"
        assert result[0]["speaker"] == "Alice"


class TestCosineSimilarity:
    """Test the cosine similarity helper."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vector returns similarity 0.0."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)


class TestListEnrolledSpeakers:
    """Test list_enrolled_speakers function."""

    def test_lists_enrolled_names(self, registry_config: dict, profiles_dir: Path):
        """Returns sorted list of enrolled speaker names."""
        np.save(profiles_dir / "Charlie.npy", np.array([0.0]))
        np.save(profiles_dir / "Alice.npy", np.array([0.0]))
        np.save(profiles_dir / "Bob.npy", np.array([0.0]))

        result = list_enrolled_speakers(registry_config)

        assert result == ["Alice", "Bob", "Charlie"]

    def test_empty_profiles_dir(self, registry_config: dict, profiles_dir: Path):
        """Returns empty list when no profiles exist."""
        result = list_enrolled_speakers(registry_config)
        assert result == []

    def test_ignores_non_npy_files(self, registry_config: dict, profiles_dir: Path):
        """Only .npy files are listed."""
        np.save(profiles_dir / "Alice.npy", np.array([0.0]))
        (profiles_dir / "notes.txt").write_text("not a profile")

        result = list_enrolled_speakers(registry_config)
        assert result == ["Alice"]


class TestDeleteSpeaker:
    """Test delete_speaker function."""

    def test_deletes_existing_profile(self, registry_config: dict, profiles_dir: Path):
        """Deleting an existing profile removes the file."""
        profile_path = profiles_dir / "Alice.npy"
        np.save(profile_path, np.array([0.0]))
        assert profile_path.exists()

        delete_speaker("Alice", registry_config)

        assert not profile_path.exists()

    def test_raises_on_nonexistent_profile(self, registry_config: dict, profiles_dir: Path):
        """Deleting a non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Speaker profile not found"):
            delete_speaker("NonExistent", registry_config)

    def test_delete_updates_list(self, registry_config: dict, profiles_dir: Path):
        """After deletion, the speaker no longer appears in list."""
        np.save(profiles_dir / "Alice.npy", np.array([0.0]))
        np.save(profiles_dir / "Bob.npy", np.array([0.0]))

        delete_speaker("Alice", registry_config)

        assert list_enrolled_speakers(registry_config) == ["Bob"]
