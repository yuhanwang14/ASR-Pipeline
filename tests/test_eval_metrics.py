"""Tests for ASR evaluation metrics."""

import json

import pytest

from src.eval_metrics import (
    compute_cer,
    compute_cpwer,
    compute_wer,
    load_pipeline_output,
    normalize_for_eval,
    parse_reference_transcript,
)


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


class TestNormalizeForEval:

    def test_strips_punctuation(self):
        assert normalize_for_eval("Hello, world!") == "hello world"

    def test_strips_chinese_punctuation(self):
        result = normalize_for_eval("嗯，对。")
        assert "，" not in result
        assert "。" not in result

    def test_preserves_chinese_characters(self):
        assert "今" in normalize_for_eval("今天讨论")

    def test_collapses_whitespace(self):
        assert normalize_for_eval("hello   world") == "hello world"

    def test_strips_ellipsis(self):
        assert "..." not in normalize_for_eval("就是... 主要的话")


class TestComputeCer:

    def test_identical(self):
        assert compute_cer("hello", "hello") == 0.0

    def test_completely_different(self):
        assert compute_cer("abc", "xyz") > 0.0

    def test_empty_reference(self):
        assert compute_cer("", "") == 0.0

    def test_chinese_text(self):
        cer = compute_cer("今天讨论项目", "今天讨论项目")
        assert cer == 0.0

    def test_partial_match(self):
        cer = compute_cer("hello world", "hello earth")
        assert 0.0 < cer < 1.0


class TestComputeWer:

    def test_identical(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert compute_wer("hello", "goodbye") > 0.0

    def test_mixed_chinese_english(self):
        wer = compute_wer("今天讨论 project", "今天讨论 project")
        assert wer == 0.0

    def test_chinese_char_substitution(self):
        # "剧" → "拒" = 1 substitution out of 4 tokens
        wer = compute_wer("他当时剧的时候", "他当时拒的时候")
        assert 0.0 < wer < 0.5


class TestComputeCpwer:

    def test_identical_text_different_labels(self):
        ref = [{"speaker": "A", "text": "hello"}, {"speaker": "B", "text": "world"}]
        hyp = [{"speaker": "X", "text": "hello"}, {"speaker": "Y", "text": "world"}]
        result = compute_cpwer(ref, hyp)
        assert result["cpwer"] == 0.0
        # Optimal mapping should map X→A, Y→B
        assert result["mapping"]["X"] == "A"
        assert result["mapping"]["Y"] == "B"

    def test_empty_segments(self):
        result = compute_cpwer([], [])
        assert result["cpwer"] == 0.0

    def test_speaker_count_mismatch(self):
        ref = [
            {"speaker": "A", "text": "hello"},
            {"speaker": "B", "text": "world"},
        ]
        hyp = [{"speaker": "X", "text": "hello world"}]
        result = compute_cpwer(ref, hyp)
        assert "mapping" in result
        assert "cpwer" in result

    def test_per_speaker_metrics(self):
        ref = [
            {"speaker": "A", "text": "hello world"},
            {"speaker": "B", "text": "foo bar"},
        ]
        hyp = [
            {"speaker": "X", "text": "hello world"},
            {"speaker": "Y", "text": "foo bar"},
        ]
        result = compute_cpwer(ref, hyp)
        assert "A" in result["per_speaker"]
        assert result["per_speaker"]["A"]["cer"] == 0.0

    def test_nonzero_cpwer_with_errors(self):
        ref = [
            {"speaker": "A", "text": "hello world"},
            {"speaker": "B", "text": "foo bar"},
        ]
        hyp = [
            {"speaker": "X", "text": "hello earth"},
            {"speaker": "Y", "text": "foo baz"},
        ]
        result = compute_cpwer(ref, hyp)
        assert result["cpwer"] > 0.0
