"""Tests for ASR evaluation metrics."""

import json

import pytest

from src.eval_metrics import load_pipeline_output, parse_reference_transcript


class TestParseReferenceTranscript:

    def test_basic_parsing(self):
        md = (
            "**Transcript**\n\n"
            "**Speaker 1:** Hello.\n\n"
            "**Speaker 2:** Hello, hello. 听得到吗？\n\n"
            "**Speaker 1:** 对。\n"
        )
        segments = parse_reference_transcript(md)
        assert len(segments) == 3
        assert segments[0] == {"speaker": "Speaker 1", "text": "Hello."}
        assert segments[1] == {"speaker": "Speaker 2", "text": "Hello, hello. 听得到吗？"}
        assert segments[2] == {"speaker": "Speaker 1", "text": "对。"}

    def test_skips_timestamp_markers(self):
        md = (
            "**Speaker 1:** Hello.\n\n"
            "(3:00)\n"
            "**Speaker 2:** World.\n"
        )
        segments = parse_reference_transcript(md)
        assert len(segments) == 2

    def test_skips_header_line(self):
        md = "**Transcript**\n\n**Speaker 1:** Hello.\n"
        segments = parse_reference_transcript(md)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker 1"

    def test_empty_input(self):
        assert parse_reference_transcript("") == []
        assert parse_reference_transcript("  \n\n  ") == []


class TestLoadPipelineOutput:

    def test_loads_segments(self, tmp_path):
        data = {
            "segments": [
                {"speaker": "SPEAKER_00", "text": "hello", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "text": "world", "start": 1.0, "end": 2.0},
            ]
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        segments = load_pipeline_output(path)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_filters_empty_text(self, tmp_path):
        data = {
            "segments": [
                {"speaker": "S0", "text": "hello"},
                {"speaker": "S0", "text": ""},
                {"speaker": "S0", "text": "   "},
            ]
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        segments = load_pipeline_output(path)
        assert len(segments) == 1

    def test_filters_think_block_junk(self, tmp_path):
        data = {
            "segments": [
                {"speaker": "UNKNOWN", "text": "<think> Thinking Process..."},
                {"speaker": "S0", "text": "real text"},
            ]
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        segments = load_pipeline_output(path)
        assert len(segments) == 1
        assert segments[0]["text"] == "real text"
