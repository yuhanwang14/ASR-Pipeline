"""Tests for output_formatter module."""

import json
from pathlib import Path

import pytest

from src.output_formatter import (
    _format_srt_time,
    save_outputs,
    to_json,
    to_rttm,
    to_srt,
    to_txt,
)

SAMPLE_SEGMENTS = [
    {
        "start": 0.5,
        "end": 3.2,
        "speaker": "Alice",
        "text": "今天我们来讨论一下 project timeline",
    },
    {
        "start": 3.5,
        "end": 5.0,
        "speaker": "SPEAKER_01",
        "text": "OK let me pull up the schedule",
    },
    {
        "start": 5.5,
        "end": 8.0,
        "speaker": "Alice",
        "text": "我觉得 deadline 可以往后推一周",
    },
]

SAMPLE_RESULT = {
    "segments": SAMPLE_SEGMENTS,
    "audio_duration_seconds": 10.0,
    "total_time_seconds": 2.5,
}

SAMPLE_METADATA = {
    "audio_file": "meeting.wav",
    "duration_seconds": 10.0,
    "processing_time_seconds": 2.5,
}


class TestToJson:
    """Tests for to_json()."""

    def test_valid_json(self):
        """to_json produces valid JSON."""
        output = to_json(SAMPLE_RESULT, SAMPLE_METADATA)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_schema_structure(self):
        """to_json output matches expected schema."""
        output = to_json(SAMPLE_RESULT, SAMPLE_METADATA)
        parsed = json.loads(output)

        assert "metadata" in parsed
        assert "speakers" in parsed
        assert "segments" in parsed
        meta = parsed["metadata"]
        assert meta["audio_file"] == "meeting.wav"
        assert meta["duration_seconds"] == 10.0
        assert meta["num_speakers"] == 2  # Alice and SPEAKER_01
        assert meta["processing_time_seconds"] == 2.5

    def test_speakers_list(self):
        """to_json includes correct speakers."""
        output = to_json(SAMPLE_RESULT, SAMPLE_METADATA)
        parsed = json.loads(output)

        speakers = parsed["speakers"]
        assert len(speakers) == 2

        alice = speakers[0]
        assert alice["id"] == "Alice"
        assert alice["name"] == "Alice"  # Not a SPEAKER_ prefix

        speaker_01 = speakers[1]
        assert speaker_01["id"] == "SPEAKER_01"
        assert speaker_01["name"] is None  # SPEAKER_ prefix -> null name

    def test_segments_preserved(self):
        """to_json preserves all segment data."""
        output = to_json(SAMPLE_RESULT, SAMPLE_METADATA)
        parsed = json.loads(output)

        segments = parsed["segments"]
        assert len(segments) == 3
        assert segments[0]["start"] == 0.5
        assert segments[0]["end"] == 3.2
        assert segments[0]["speaker"] == "Alice"
        assert segments[0]["text"] == "今天我们来讨论一下 project timeline"

    def test_chinese_text_not_escaped(self):
        """to_json uses ensure_ascii=False so Chinese characters are not escaped."""
        output = to_json(SAMPLE_RESULT, SAMPLE_METADATA)
        assert "今天" in output
        assert "\\u" not in output



class TestToSrt:
    """Tests for to_srt()."""

    def test_correct_format(self):
        """to_srt produces correct SRT format with numbered entries."""
        output = to_srt(SAMPLE_SEGMENTS)
        lines = output.split("\n")

        # First entry
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,500 --> 00:00:03,200"
        assert lines[2] == "[Alice] 今天我们来讨论一下 project timeline"
        assert lines[3] == ""  # blank separator

    def test_timestamps(self):
        """to_srt produces correct timestamps."""
        output = to_srt(SAMPLE_SEGMENTS)
        assert "00:00:00,500 --> 00:00:03,200" in output
        assert "00:00:03,500 --> 00:00:05,000" in output
        assert "00:00:05,500 --> 00:00:08,000" in output

    def test_chinese_english_text(self):
        """to_srt handles Chinese-English mixed text correctly."""
        output = to_srt(SAMPLE_SEGMENTS)
        assert "今天我们来讨论一下 project timeline" in output
        assert "我觉得 deadline 可以往后推一周" in output

    def test_sequential_numbering(self):
        """to_srt uses sequential entry numbers starting at 1."""
        output = to_srt(SAMPLE_SEGMENTS)
        lines = output.split("\n")
        assert lines[0] == "1"
        assert lines[4] == "2"
        assert lines[8] == "3"

    def test_empty_segments(self):
        """to_srt returns empty string for empty segments."""
        output = to_srt([])
        assert output == ""


class TestToRttm:
    """Tests for to_rttm()."""

    def test_correct_format(self):
        """to_rttm produces correct RTTM format."""
        output = to_rttm(SAMPLE_SEGMENTS, "meeting.wav")
        lines = output.strip().split("\n")
        assert len(lines) == 3

        parts = lines[0].split()
        assert parts[0] == "SPEAKER"
        assert parts[1] == "meeting.wav"
        assert parts[2] == "1"
        assert float(parts[3]) == pytest.approx(0.5)
        assert float(parts[4]) == pytest.approx(2.7)  # 3.2 - 0.5
        assert parts[5] == "<NA>"
        assert parts[6] == "<NA>"
        assert parts[7] == "Alice"
        assert parts[8] == "<NA>"
        assert parts[9] == "<NA>"

    def test_duration_calculation(self):
        """to_rttm calculates duration correctly."""
        output = to_rttm(SAMPLE_SEGMENTS, "test.wav")
        lines = output.strip().split("\n")

        # Second segment: 5.0 - 3.5 = 1.5
        parts = lines[1].split()
        assert float(parts[4]) == pytest.approx(1.5)

        # Third segment: 8.0 - 5.5 = 2.5
        parts = lines[2].split()
        assert float(parts[4]) == pytest.approx(2.5)

    def test_audio_filename_in_output(self):
        """to_rttm uses the provided audio filename."""
        output = to_rttm(SAMPLE_SEGMENTS, "custom_audio.mp3")
        assert "custom_audio.mp3" in output


class TestToTxt:
    """Tests for to_txt()."""

    def test_correct_format(self):
        """to_txt produces correct plain text format."""
        output = to_txt(SAMPLE_SEGMENTS)
        lines = output.split("\n")
        assert len(lines) == 3

        assert lines[0] == "[00:00:00] Alice: 今天我们来讨论一下 project timeline"
        assert lines[1] == "[00:00:03] SPEAKER_01: OK let me pull up the schedule"
        assert lines[2] == "[00:00:05] Alice: 我觉得 deadline 可以往后推一周"

    def test_timestamp_format(self):
        """to_txt uses HH:MM:SS format for timestamps."""
        output = to_txt(SAMPLE_SEGMENTS)
        assert "[00:00:00]" in output
        assert "[00:00:03]" in output
        assert "[00:00:05]" in output

    def test_empty_segments(self):
        """to_txt returns empty string for empty segments."""
        output = to_txt([])
        assert output == ""


class TestFormatSrtTime:
    """Tests for _format_srt_time()."""

    def test_zero(self):
        """Handles zero seconds."""
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_fractional_seconds(self):
        """Handles fractional seconds."""
        assert _format_srt_time(1.5) == "00:00:01,500"
        assert _format_srt_time(0.123) == "00:00:00,123"

    def test_minutes(self):
        """Handles minutes correctly."""
        assert _format_srt_time(65.0) == "00:01:05,000"

    def test_hours(self):
        """Handles hours correctly."""
        assert _format_srt_time(3661.5) == "01:01:01,500"

    def test_large_values(self):
        """Handles large values (multi-hour recordings)."""
        assert _format_srt_time(36000.0) == "10:00:00,000"

    def test_millisecond_precision(self):
        """Preserves millisecond precision."""
        assert _format_srt_time(1.001) == "00:00:01,001"
        assert _format_srt_time(1.999) == "00:00:01,999"

    def test_negative_clamped_to_zero(self):
        """Negative values are clamped to zero."""
        assert _format_srt_time(-5.0) == "00:00:00,000"


class TestSaveOutputs:
    """Tests for save_outputs()."""

    def test_saves_all_configured_formats(self, tmp_output_dir):
        """save_outputs creates files for all configured formats."""
        config = {
            "output": {
                "formats": ["json", "srt", "rttm", "txt"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")

        assert len(saved) == 4
        for path in saved:
            assert Path(path).exists()

        extensions = {Path(p).suffix for p in saved}
        assert extensions == {".json", ".srt", ".rttm", ".txt"}

    def test_saves_subset_of_formats(self, tmp_output_dir):
        """save_outputs only creates files for configured formats."""
        config = {
            "output": {
                "formats": ["json", "txt"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")

        assert len(saved) == 2
        extensions = {Path(p).suffix for p in saved}
        assert extensions == {".json", ".txt"}

    def test_json_content_is_valid(self, tmp_output_dir):
        """Saved JSON file contains valid JSON."""
        config = {
            "output": {
                "formats": ["json"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")
        content = Path(saved[0]).read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "segments" in parsed
        assert "metadata" in parsed

    def test_srt_content_has_entries(self, tmp_output_dir):
        """Saved SRT file has correct number of entries."""
        config = {
            "output": {
                "formats": ["srt"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")
        content = Path(saved[0]).read_text(encoding="utf-8")
        # Each entry has: number, timestamp, text, blank line
        assert "1\n" in content
        assert "2\n" in content
        assert "3\n" in content

    def test_output_filename_from_audio(self, tmp_output_dir):
        """Output files use the audio file stem as the base name."""
        config = {
            "output": {
                "formats": ["txt"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "/path/to/my_recording.mp4")
        assert Path(saved[0]).name == "my_recording.txt"

    def test_chinese_text_preserved_in_files(self, tmp_output_dir):
        """Chinese text is correctly written to output files."""
        config = {
            "output": {
                "formats": ["txt", "srt", "json"],
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")
        for path in saved:
            content = Path(path).read_text(encoding="utf-8")
            assert "今天" in content

    def test_unknown_format_skipped(self, tmp_output_dir):
        """Unknown formats are skipped without error."""
        config = {
            "output": {
                "formats": ["json", "pdf"],  # pdf is unsupported
                "output_dir": str(tmp_output_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")
        assert len(saved) == 1
        assert Path(saved[0]).suffix == ".json"

    def test_creates_output_dir_if_missing(self, tmp_output_dir):
        """save_outputs creates the output directory if it does not exist."""
        nested_dir = tmp_output_dir / "nested" / "deep"
        config = {
            "output": {
                "formats": ["txt"],
                "output_dir": str(nested_dir),
            }
        }

        saved = save_outputs(SAMPLE_RESULT, config, "meeting.wav")
        assert len(saved) == 1
        assert nested_dir.exists()
