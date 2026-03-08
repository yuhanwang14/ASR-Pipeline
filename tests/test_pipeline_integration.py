"""Integration tests for the full pipeline with mocked backends."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a minimal audio file path for testing (audio loading is mocked)."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 100)  # Dummy file so path exists
    return str(audio_path)


@pytest.fixture
def mock_config(tmp_path):
    """Pipeline config with output_dir pointing to tmp_path."""
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
            "speaker_profiles_dir": str(tmp_path / "speaker_profiles"),
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
            "backend": "llama-cpp",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "tasks": {
                "speaker_correction": True,
                "text_correction": True,
                "summarization": True,
            },
        },
        "output": {
            "formats": ["json", "srt", "txt"],
            "output_dir": str(tmp_path / "output"),
            "save_intermediate": True,
        },
    }


@pytest.fixture
def sample_segments():
    """Segments returned by diarization and passed through subsequent stages."""
    return [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "Hello world"},
        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi there"},
    ]


@pytest.fixture
def sample_vad_result():
    """Stage 0 result dict as returned by _run_stage_0."""
    return {
        "speech_segments": [{"start": 0.0, "end": 2.0}],
        "timestamp_map": [
            {
                "original_start": 0.0,
                "original_end": 2.0,
                "clean_start": 0.0,
                "clean_end": 2.0,
            }
        ],
        "clean_waveform_samples": 32000,
    }


@pytest.fixture
def sample_llm_result(sample_segments):
    """Return value for mocked run_llm_postprocess."""
    return {
        "segments": sample_segments,
        "summary": "Test meeting summary.",
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end pipeline tests with all external dependencies mocked."""

    def test_full_pipeline_with_mocked_backends(
        self,
        tmp_path,
        mock_audio_file,
        mock_config,
        sample_vad_result,
        sample_segments,
        sample_llm_result,
    ):
        """Run run_pipeline() with every stage mocked, verify outputs."""
        from src.pipeline import run_pipeline

        output_dir = str(tmp_path / "output")
        mock_config["output"]["output_dir"] = output_dir

        with (
            patch(
                "src.audio_preprocessing.load_audio",
                return_value=(torch.randn(1, 32000), 16000),
            ),
            patch("src.gpu_utils.get_vram_usage", return_value=0.0),
            patch("src.pipeline._run_stage_0", return_value=sample_vad_result),
            patch(
                "src.pipeline._build_clean_waveform",
                return_value=torch.randn(1, 32000),
            ),
            patch(
                "src.pipeline._run_stage_1",
                return_value={"segments": sample_segments},
            ),
            patch(
                "src.pipeline._run_stage_2",
                return_value={"segments": sample_segments},
            ),
            patch("src.pipeline._run_stage_3", return_value=sample_llm_result),
        ):
            result = run_pipeline(mock_audio_file, config=mock_config, output_dir=output_dir)

        # All stages should have produced output in the result dict
        assert "stages" in result
        assert "vad" in result["stages"]
        assert "diarization" in result["stages"]
        assert "transcription" in result["stages"]
        assert "llm_postprocess" in result["stages"]

        # Final segments and summary are exposed at top level
        assert "segments" in result
        assert len(result["segments"]) == 2
        assert "summary" in result
        assert result["summary"] == "Test meeting summary."

        # Timing metadata is present
        assert "timings" in result
        assert "total_time_seconds" in result
        assert result["total_time_seconds"] >= 0

        # Intermediate files were saved
        output_path = Path(output_dir)
        for stage_idx in range(4):
            assert (output_path / f"stage_{stage_idx}.json").exists(), (
                f"Intermediate file for stage_{stage_idx} not found"
            )

    def test_crash_recovery_resumes_from_stage(
        self,
        tmp_path,
        mock_audio_file,
        mock_config,
        sample_vad_result,
        sample_segments,
        sample_llm_result,
    ):
        """Pre-populate stage_0 and stage_1 intermediates, verify they are loaded from cache."""
        from src.pipeline import run_pipeline

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_config["output"]["output_dir"] = str(output_dir)

        # Pre-save stage 0 and stage 1 intermediates
        stage_0_data = sample_vad_result
        stage_1_data = {"segments": sample_segments}

        (output_dir / "stage_0.json").write_text(json.dumps(stage_0_data), encoding="utf-8")
        (output_dir / "stage_1.json").write_text(json.dumps(stage_1_data), encoding="utf-8")

        # Track which stage runners are called
        stage_0_mock = MagicMock()
        stage_1_mock = MagicMock()
        stage_2_mock = MagicMock(return_value={"segments": sample_segments})
        stage_3_mock = MagicMock(return_value=sample_llm_result)

        with (
            patch(
                "src.audio_preprocessing.load_audio",
                return_value=(torch.randn(1, 32000), 16000),
            ),
            patch("src.gpu_utils.get_vram_usage", return_value=0.0),
            patch("src.pipeline._run_stage_0", stage_0_mock),
            patch(
                "src.pipeline._build_clean_waveform",
                return_value=torch.randn(1, 32000),
            ),
            patch("src.pipeline._run_stage_1", stage_1_mock),
            patch("src.pipeline._run_stage_2", stage_2_mock),
            patch("src.pipeline._run_stage_3", stage_3_mock),
        ):
            result = run_pipeline(
                mock_audio_file,
                config=mock_config,
                output_dir=str(output_dir),
            )

        # Stages 0 and 1 should NOT have been called (loaded from cache)
        stage_0_mock.assert_not_called()
        stage_1_mock.assert_not_called()

        # Stages 2 and 3 should have been called
        stage_2_mock.assert_called_once()
        stage_3_mock.assert_called_once()

        # Cached data should appear in result
        assert result["stages"]["vad"] == stage_0_data
        assert result["stages"]["diarization"] == stage_1_data

    def test_stage_1_receives_clean_waveform_and_remaps(
        self,
        tmp_path,
        mock_audio_file,
        mock_config,
        sample_segments,
        sample_llm_result,
    ):
        """Verify that Stage 1 is called with clean waveform and timestamp_map."""
        from src.pipeline import run_pipeline

        output_dir = str(tmp_path / "output")
        mock_config["output"]["output_dir"] = output_dir

        stage_0_result = {
            "speech_segments": [{"start": 1.0, "end": 3.0}],
            "timestamp_map": [
                {
                    "original_start": 1.0,
                    "original_end": 3.0,
                    "clean_start": 0.0,
                    "clean_end": 2.0,
                }
            ],
            "clean_waveform_samples": 32000,
        }

        stage_1_mock = MagicMock(return_value={"segments": sample_segments})
        clean_wf = torch.randn(1, 32000)

        with (
            patch(
                "src.audio_preprocessing.load_audio",
                return_value=(torch.randn(1, 48000), 16000),
            ),
            patch("src.gpu_utils.get_vram_usage", return_value=0.0),
            patch("src.pipeline._run_stage_0", return_value=stage_0_result),
            patch("src.pipeline._build_clean_waveform", return_value=clean_wf),
            patch("src.pipeline._run_stage_1", stage_1_mock),
            patch(
                "src.pipeline._run_stage_2",
                return_value={"segments": sample_segments},
            ),
            patch("src.pipeline._run_stage_3", return_value=sample_llm_result),
        ):
            run_pipeline(mock_audio_file, config=mock_config, output_dir=output_dir)

        # Stage 1 should receive clean_waveform and timestamp_map
        call_args = stage_1_mock.call_args
        assert torch.equal(call_args[0][0], clean_wf)
        assert call_args[0][1] == 16000
        assert call_args[0][3] == stage_0_result["timestamp_map"]

    def test_stage_2_receives_segments_directly(
        self,
        tmp_path,
        mock_audio_file,
        mock_config,
        sample_segments,
        sample_llm_result,
    ):
        """Verify that Stage 2 receives segments from Stage 1 directly (not from disk)."""
        from src.pipeline import run_pipeline

        output_dir = str(tmp_path / "output")
        mock_config["output"]["output_dir"] = output_dir

        stage_0_result = {
            "speech_segments": [{"start": 0.0, "end": 2.0}],
            "timestamp_map": [
                {
                    "original_start": 0.0,
                    "original_end": 2.0,
                    "clean_start": 0.0,
                    "clean_end": 2.0,
                }
            ],
            "clean_waveform_samples": 32000,
        }

        stage_1_segments = [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]

        stage_2_mock = MagicMock(return_value={"segments": sample_segments})

        with (
            patch(
                "src.audio_preprocessing.load_audio",
                return_value=(torch.randn(1, 32000), 16000),
            ),
            patch("src.gpu_utils.get_vram_usage", return_value=0.0),
            patch("src.pipeline._run_stage_0", return_value=stage_0_result),
            patch(
                "src.pipeline._build_clean_waveform",
                return_value=torch.randn(1, 32000),
            ),
            patch(
                "src.pipeline._run_stage_1",
                return_value={"segments": stage_1_segments},
            ),
            patch("src.pipeline._run_stage_2", stage_2_mock),
            patch("src.pipeline._run_stage_3", return_value=sample_llm_result),
        ):
            run_pipeline(mock_audio_file, config=mock_config, output_dir=output_dir)

        # Stage 2 should receive segments directly from stage 1, not from disk
        call_args = stage_2_mock.call_args
        assert call_args[0][2] == stage_1_segments

    def test_pipeline_produces_all_output_formats(
        self,
        tmp_path,
        mock_audio_file,
        mock_config,
        sample_segments,
        sample_llm_result,
    ):
        """Run pipeline then save_outputs, verify JSON/SRT/TXT files exist."""
        from src.output_formatter import save_outputs

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a pipeline result dict as run_pipeline would return
        result = {
            "audio_file": mock_audio_file,
            "output_dir": str(output_dir),
            "stages": {
                "vad": {},
                "diarization": {},
                "transcription": {},
                "llm_postprocess": sample_llm_result,
            },
            "segments": sample_segments,
            "summary": "Test summary.",
            "warnings": [],
            "timings": {"audio_load": 0.1, "stage_0": 0.2},
            "total_time_seconds": 0.3,
            "audio_duration_seconds": 2.0,
        }

        # Configure output to include all formats
        mock_config["output"]["formats"] = ["json", "srt", "txt"]
        mock_config["output"]["output_dir"] = str(output_dir)

        saved = save_outputs(result, mock_config, mock_audio_file)

        # Verify files exist
        assert len(saved) == 3
        extensions = {Path(p).suffix for p in saved}
        assert ".json" in extensions
        assert ".srt" in extensions
        assert ".txt" in extensions

        # Verify each file has content
        for path in saved:
            content = Path(path).read_text(encoding="utf-8")
            assert len(content) > 0, f"Output file is empty: {path}"

        # Verify JSON content is valid and contains expected fields
        json_path = [p for p in saved if p.endswith(".json")][0]
        json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert "metadata" in json_data
        assert "segments" in json_data
        assert len(json_data["segments"]) == 2

        # Verify SRT content has correct format
        srt_path = [p for p in saved if p.endswith(".srt")][0]
        srt_content = Path(srt_path).read_text(encoding="utf-8")
        assert "1\n" in srt_content
        assert "-->" in srt_content
        assert "[SPEAKER_00]" in srt_content
