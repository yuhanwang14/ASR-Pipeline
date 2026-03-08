"""Tests for LLM post-processing module.

Covers:
- Pure function tests (format, parse, prompt builders)
- TPST safety check tests (critical for preventing hallucination)
- JSON text correction tests (parse, validate, apply)
- Integration tests with a mock backend
"""

import pytest

from src.llm_postprocess import (
    LLMBackend,
    _extract_words,
    apply_speaker_corrections,
    apply_text_corrections,
    build_speaker_correction_prompt,
    build_text_correction_prompt,
    format_diarization_lm,
    parse_diarization_lm,
    parse_json_corrections,
    run_llm_postprocess,
    tpst_check,
    validate_text_correction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLMBackend:
    """Mock backend that returns predefined text for each generate() call."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self._call_idx = 0
        self.loaded = False
        self.unloaded = False
        self.generate_calls: list[str] = []

    def load(self) -> None:
        self.loaded = True

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        self.generate_calls.append(prompt)
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return ""

    def unload(self) -> None:
        self.unloaded = True


# Verify MockLLMBackend satisfies the protocol
assert isinstance(MockLLMBackend(), LLMBackend)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_segments() -> list[dict]:
    """Two-speaker English segments."""
    return [
        {"speaker": "SPEAKER_00", "text": "Hello world"},
        {"speaker": "SPEAKER_01", "text": "Hi there"},
    ]


@pytest.fixture
def chinese_english_segments() -> list[dict]:
    """Mixed Chinese-English (code-switching) segments."""
    return [
        {"speaker": "SPEAKER_00", "text": "今天我们来讨论 project timeline"},
        {"speaker": "SPEAKER_01", "text": "OK let me pull up the schedule"},
        {"speaker": "SPEAKER_00", "text": "我觉得 deadline 可以往后推一周"},
    ]


@pytest.fixture
def llm_config() -> dict:
    """Minimal config for LLM post-processing tests."""
    return {
        "llm": {
            "model_path": "/fake/model.gguf",
            "model_id": "Qwen/Qwen3.5-9B",
            "backend": "llama-cpp",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "tasks": {
                "speaker_correction": True,
                "text_correction": True,
            },
        }
    }


# ===================================================================
# Pure function tests
# ===================================================================


class TestFormatDiarizationLM:
    """Test format_diarization_lm."""

    def test_basic_formatting(self, simple_segments):
        result = format_diarization_lm(simple_segments)
        lines = result.strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "<speaker:SPEAKER_00> Hello world"
        assert lines[1] == "<speaker:SPEAKER_01> Hi there"

    def test_chinese_english_formatting(self, chinese_english_segments):
        result = format_diarization_lm(chinese_english_segments)
        lines = result.strip().splitlines()
        assert len(lines) == 3
        assert "<speaker:SPEAKER_00>" in lines[0]
        assert "今天我们来讨论 project timeline" in lines[0]

    def test_empty_segments(self):
        result = format_diarization_lm([])
        assert result == ""

    def test_missing_speaker_key(self):
        segments = [{"text": "some text"}]
        result = format_diarization_lm(segments)
        assert "<speaker:UNKNOWN>" in result

    def test_missing_text_key(self):
        segments = [{"speaker": "SPEAKER_00"}]
        result = format_diarization_lm(segments)
        assert "<speaker:SPEAKER_00>" in result


class TestParseDiarizationLM:
    """Test parse_diarization_lm."""

    def test_basic_parse(self):
        text = "<speaker:SPEAKER_00> Hello world\n<speaker:SPEAKER_01> Hi there"
        result = parse_diarization_lm(text)
        assert len(result) == 2
        assert result[0] == {"speaker": "SPEAKER_00", "text": "Hello world"}
        assert result[1] == {"speaker": "SPEAKER_01", "text": "Hi there"}

    def test_roundtrip_with_format(self, simple_segments):
        formatted = format_diarization_lm(simple_segments)
        parsed = parse_diarization_lm(formatted)
        assert len(parsed) == len(simple_segments)
        for orig, recovered in zip(simple_segments, parsed, strict=True):
            assert orig["speaker"] == recovered["speaker"]
            assert orig["text"] == recovered["text"]

    def test_roundtrip_chinese_english(self, chinese_english_segments):
        formatted = format_diarization_lm(chinese_english_segments)
        parsed = parse_diarization_lm(formatted)
        assert len(parsed) == len(chinese_english_segments)
        for orig, recovered in zip(chinese_english_segments, parsed, strict=True):
            assert orig["speaker"] == recovered["speaker"]
            assert orig["text"] == recovered["text"]

    def test_empty_input(self):
        result = parse_diarization_lm("")
        assert result == []

    def test_whitespace_only(self):
        result = parse_diarization_lm("   \n  \n  ")
        assert result == []

    def test_line_without_speaker_tag(self):
        text = "plain text without speaker tag"
        result = parse_diarization_lm(text)
        assert len(result) == 1
        assert result[0]["speaker"] == "UNKNOWN"

    def test_extra_whitespace_around_lines(self):
        text = "  <speaker:SPEAKER_00> Hello world  \n  <speaker:SPEAKER_01> Hi  "
        result = parse_diarization_lm(text)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert result[1]["text"] == "Hi"


class TestBuildSpeakerCorrectionPrompt:
    """Test build_speaker_correction_prompt (JSON output format)."""

    def test_requests_json_output(self):
        prompt = build_speaker_correction_prompt("dummy")
        assert "JSON" in prompt
        assert "[]" in prompt

    def test_includes_json_example(self):
        prompt = build_speaker_correction_prompt("test")
        assert '"old_speaker"' in prompt
        assert '"new_speaker"' in prompt
        assert '"line"' in prompt

    def test_includes_transcript(self):
        transcript = "<speaker:SPEAKER_00> Hello\n<speaker:SPEAKER_01> World"
        prompt = build_speaker_correction_prompt(transcript)
        assert transcript in prompt

    def test_uses_chatml_format(self):
        prompt = build_speaker_correction_prompt("test")
        assert "<|im_start|>system" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_includes_few_shot_example(self):
        prompt = build_speaker_correction_prompt("test")
        assert "Example input:" in prompt
        assert "Example output:" in prompt

    def test_includes_think_block(self):
        prompt = build_speaker_correction_prompt("test")
        assert "<think>\n</think>" in prompt


class TestBuildTextCorrectionPrompt:
    """Test build_text_correction_prompt (JSON output format)."""

    def test_requests_json_output(self):
        prompt = build_text_correction_prompt("dummy")
        assert "JSON" in prompt
        assert "[]" in prompt

    def test_contains_error_examples(self):
        prompt = build_text_correction_prompt("test")
        assert "sync" in prompt or "demo" in prompt or "VC" in prompt

    def test_includes_json_example(self):
        prompt = build_text_correction_prompt("test")
        assert '"line"' in prompt
        assert '"original"' in prompt
        assert '"corrected"' in prompt

    def test_includes_transcript(self):
        transcript = "<speaker:SPEAKER_00> Hello\n<speaker:SPEAKER_01> World"
        prompt = build_text_correction_prompt(transcript)
        assert transcript in prompt

    def test_uses_chatml_format(self):
        prompt = build_text_correction_prompt("test")
        assert "<|im_start|>system" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_includes_think_block(self):
        prompt = build_text_correction_prompt("test")
        assert "<think>\n</think>" in prompt


# ===================================================================
# JSON text correction tests
# ===================================================================


class TestParseJsonCorrections:
    """Test parse_json_corrections."""

    def test_valid_array(self):
        raw = '[{"line": 0, "original": "think", "corrected": "sync"}]'
        result = parse_json_corrections(raw)
        assert len(result) == 1
        assert result[0]["original"] == "think"

    def test_empty_array(self):
        assert parse_json_corrections("[]") == []

    def test_no_corrections(self):
        assert parse_json_corrections("") == []

    def test_strips_preamble(self):
        raw = 'Here are the corrections:\n[{"line": 0, "original": "a", "corrected": "b"}]'
        result = parse_json_corrections(raw)
        assert len(result) == 1

    def test_strips_trailing_text(self):
        raw = '[{"line": 0, "original": "a", "corrected": "b"}]\nDone!'
        result = parse_json_corrections(raw)
        assert len(result) == 1

    def test_malformed_json(self):
        assert parse_json_corrections("[{broken") == []

    def test_not_an_array(self):
        assert parse_json_corrections('{"line": 0}') == []

    def test_multiple_corrections(self):
        raw = '[{"line": 0, "original": "a", "corrected": "b"}, {"line": 2, "original": "c", "corrected": "d"}]'
        result = parse_json_corrections(raw)
        assert len(result) == 2


class TestValidateTextCorrection:
    """Test validate_text_correction."""

    def test_valid_correction(self):
        ok, _ = validate_text_correction(
            "today think about it", {"original": "think", "corrected": "sync"}
        )
        assert ok is True

    def test_original_not_in_segment(self):
        ok, reason = validate_text_correction(
            "hello world", {"original": "banana", "corrected": "apple"}
        )
        assert ok is False
        assert "not found" in reason

    def test_empty_original(self):
        ok, _ = validate_text_correction("hello", {"original": "", "corrected": "x"})
        assert ok is False

    def test_empty_corrected(self):
        ok, _ = validate_text_correction("hello", {"original": "hello", "corrected": ""})
        assert ok is False

    def test_no_change(self):
        ok, _ = validate_text_correction("hello", {"original": "hello", "corrected": "hello"})
        assert ok is False

    def test_suspiciously_long_correction(self):
        ok, reason = validate_text_correction("hi", {"original": "hi", "corrected": "a" * 100})
        assert ok is False
        assert "long" in reason.lower()


class TestApplyTextCorrections:
    """Test apply_text_corrections."""

    def test_applies_valid_correction(self):
        segments = [{"speaker": "S0", "text": "today think about it", "start": 0.0, "end": 1.0}]
        corrections = [{"line": 0, "original": "think", "corrected": "sync"}]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "today sync about it"
        assert result[0]["start"] == 0.0
        assert warnings == []

    def test_skips_invalid_line(self):
        segments = [{"speaker": "S0", "text": "hello"}]
        corrections = [{"line": 5, "original": "x", "corrected": "y"}]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "hello"
        assert len(warnings) == 1

    def test_skips_missing_original(self):
        segments = [{"speaker": "S0", "text": "hello world"}]
        corrections = [{"line": 0, "original": "banana", "corrected": "apple"}]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "hello world"
        assert len(warnings) == 1

    def test_rate_limits_corrections(self):
        segments = [{"speaker": "S0", "text": "a b c"}]
        corrections = [{"line": 0, "original": "a", "corrected": "x"}] * 25
        result, warnings = apply_text_corrections(segments, corrections, max_corrections=20)
        assert result[0]["text"] == "a b c"
        assert any("hallucination" in w.lower() for w in warnings)

    def test_empty_corrections(self):
        segments = [{"speaker": "S0", "text": "hello"}]
        result, warnings = apply_text_corrections(segments, [])
        assert result[0]["text"] == "hello"
        assert warnings == []

    def test_multiple_corrections_different_lines(self):
        segments = [
            {"speaker": "S0", "text": "think about it"},
            {"speaker": "S1", "text": "the 飞机 is good"},
        ]
        corrections = [
            {"line": 0, "original": "think", "corrected": "sync"},
            {"line": 1, "original": "飞机", "corrected": "VC"},
        ]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "sync about it"
        assert result[1]["text"] == "the VC is good"
        assert warnings == []

    def test_does_not_mutate_original(self):
        segments = [{"speaker": "S0", "text": "hello world"}]
        corrections = [{"line": 0, "original": "hello", "corrected": "hi"}]
        apply_text_corrections(segments, corrections)
        assert segments[0]["text"] == "hello world"

    def test_negative_line_number(self):
        segments = [{"speaker": "S0", "text": "hello"}]
        corrections = [{"line": -1, "original": "hello", "corrected": "hi"}]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "hello"
        assert len(warnings) == 1

    def test_non_integer_line(self):
        segments = [{"speaker": "S0", "text": "hello"}]
        corrections = [{"line": "zero", "original": "hello", "corrected": "hi"}]
        result, warnings = apply_text_corrections(segments, corrections)
        assert result[0]["text"] == "hello"
        assert len(warnings) == 1


class TestApplySpeakerCorrections:
    """Test apply_speaker_corrections."""

    def test_applies_valid_correction(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "text": "World", "start": 1.0, "end": 2.0},
        ]
        corrections = [{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}]
        result, warnings = apply_speaker_corrections(segments, corrections)
        assert result[0]["speaker"] == "SPEAKER_01"
        assert result[0]["start"] == 0.0  # metadata preserved
        assert result[1]["speaker"] == "SPEAKER_01"  # unchanged
        assert warnings == []

    def test_skips_speaker_mismatch(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        corrections = [{"line": 0, "old_speaker": "SPEAKER_99", "new_speaker": "SPEAKER_01"}]
        result, warnings = apply_speaker_corrections(segments, corrections)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert len(warnings) == 1

    def test_skips_invalid_line(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        corrections = [{"line": 5, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}]
        result, warnings = apply_speaker_corrections(segments, corrections)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert len(warnings) == 1

    def test_empty_corrections(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result, warnings = apply_speaker_corrections(segments, [])
        assert result[0]["speaker"] == "SPEAKER_00"
        assert warnings == []

    def test_does_not_mutate_original(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        corrections = [{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}]
        apply_speaker_corrections(segments, corrections)
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_missing_speaker_fields(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        corrections = [{"line": 0}]
        result, warnings = apply_speaker_corrections(segments, corrections)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert len(warnings) == 1


# ===================================================================
# TPST check tests — CRITICAL for preventing hallucination
# ===================================================================


class TestExtractWords:
    """Test the internal _extract_words helper."""

    def test_english_words(self):
        segments = [{"text": "Hello world"}]
        tokens = _extract_words(segments)
        assert tokens == ["hello", "world"]

    def test_cjk_characters(self):
        segments = [{"text": "你好世界"}]
        tokens = _extract_words(segments)
        assert tokens == ["你", "好", "世", "界"]

    def test_mixed_cjk_english(self):
        segments = [{"text": "今天讨论 project"}]
        tokens = _extract_words(segments)
        assert tokens == ["今", "天", "讨", "论", "project"]

    def test_strips_punctuation(self):
        segments = [{"text": "Hello, world! How are you?"}]
        tokens = _extract_words(segments)
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_strips_speaker_tags(self):
        segments = [{"text": "<speaker:SPEAKER_00> Hello world"}]
        tokens = _extract_words(segments)
        assert tokens == ["hello", "world"]

    def test_case_insensitive(self):
        segments = [{"text": "Hello WORLD"}]
        tokens = _extract_words(segments)
        assert tokens == ["hello", "world"]


class TestTPSTCheck:
    """Test tpst_check — the safety mechanism against LLM hallucination."""

    def test_passes_when_only_speaker_labels_changed(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
            {"speaker": "SPEAKER_01", "text": "Hi there"},
        ]
        corrected = [
            {"speaker": "SPEAKER_01", "text": "Hello world"},
            {"speaker": "SPEAKER_00", "text": "Hi there"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == corrected
        assert warnings == []

    def test_passes_when_only_punctuation_changed(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "Hello, world!"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == corrected
        assert warnings == []

    def test_fails_when_word_added(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "Hello beautiful world"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == original
        assert len(warnings) > 0
        assert any("TPST" in w for w in warnings)

    def test_fails_when_word_removed(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello beautiful world"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == original
        assert len(warnings) > 0
        assert any("TPST" in w for w in warnings)

    def test_fails_when_word_modified(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "Hello earth"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == original
        assert len(warnings) > 0
        assert any("modified" in w.lower() or "mismatch" in w.lower() for w in warnings)

    def test_passes_with_mixed_chinese_english_only_speakers_changed(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "今天我们来讨论 project timeline"},
            {"speaker": "SPEAKER_01", "text": "OK let me pull up the schedule"},
        ]
        corrected = [
            {"speaker": "SPEAKER_01", "text": "今天我们来讨论 project timeline"},
            {"speaker": "SPEAKER_00", "text": "OK let me pull up the schedule"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == corrected
        assert warnings == []

    def test_fails_when_chinese_character_changed(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "今天讨论"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "明天讨论"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == original
        assert len(warnings) > 0

    def test_fails_when_chinese_character_added(self):
        original = [
            {"speaker": "SPEAKER_00", "text": "今天讨论"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "今天来讨论"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == original
        assert len(warnings) > 0

    def test_handles_empty_segments(self):
        result, warnings = tpst_check([], [])
        assert result == []
        assert warnings == []

    def test_handles_original_empty_corrected_not(self):
        corrected = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result, warnings = tpst_check([], corrected)
        assert result == []
        assert len(warnings) > 0

    def test_handles_corrected_empty_original_not(self):
        original = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result, warnings = tpst_check(original, [])
        assert result == original
        assert len(warnings) > 0

    def test_passes_with_only_punctuation_and_speaker_changes(self):
        """Both speaker labels AND punctuation changed — should still pass."""
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
            {"speaker": "SPEAKER_01", "text": "Hi there"},
        ]
        corrected = [
            {"speaker": "SPEAKER_01", "text": "Hello, world."},
            {"speaker": "SPEAKER_00", "text": "Hi there!"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == corrected
        assert warnings == []

    def test_single_segment_no_change(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result, warnings = tpst_check(segments, segments)
        assert result == segments
        assert warnings == []

    def test_segments_merged_but_words_same(self):
        """If segments are merged/split but the total words are identical, it passes."""
        original = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_00", "text": "world"},
        ]
        corrected = [
            {"speaker": "SPEAKER_00", "text": "Hello world"},
        ]
        result, warnings = tpst_check(original, corrected)
        assert result == corrected
        assert warnings == []


# ===================================================================
# Integration tests with mock backend
# ===================================================================


class TestRunLLMPostprocess:
    """Test run_llm_postprocess with MockLLMBackend."""

    def test_calls_load_and_unload(self, simple_segments, llm_config):
        backend = MockLLMBackend(
            responses=[
                # speaker correction response (JSON, swap speakers)
                '[{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}, '
                '{"line": 1, "old_speaker": "SPEAKER_01", "new_speaker": "SPEAKER_00"}]',
                # text correction response (JSON, no errors)
                "[]",
            ]
        )
        run_llm_postprocess(simple_segments, llm_config, backend=backend)
        assert backend.loaded is True
        assert backend.unloaded is True

    def test_runs_all_enabled_tasks(self, simple_segments, llm_config):
        backend = MockLLMBackend(
            responses=[
                # speaker correction (JSON, swap speakers)
                '[{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}, '
                '{"line": 1, "old_speaker": "SPEAKER_01", "new_speaker": "SPEAKER_00"}]',
                # text correction (JSON, no errors)
                "[]",
            ]
        )
        run_llm_postprocess(simple_segments, llm_config, backend=backend)
        assert len(backend.generate_calls) == 2

    def test_skips_disabled_tasks(self, simple_segments, llm_config):
        llm_config["llm"]["tasks"]["speaker_correction"] = False
        llm_config["llm"]["tasks"]["text_correction"] = False

        backend = MockLLMBackend(responses=[])
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert len(backend.generate_calls) == 0
        assert result["segments"] == simple_segments
        assert result["warnings"] == []

    def test_only_speaker_correction(self, simple_segments, llm_config):
        llm_config["llm"]["tasks"]["text_correction"] = False

        backend = MockLLMBackend(
            responses=[
                '[{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}, '
                '{"line": 1, "old_speaker": "SPEAKER_01", "new_speaker": "SPEAKER_00"}]',
            ]
        )
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert len(backend.generate_calls) == 1
        # Speakers should be swapped
        assert result["segments"][0]["speaker"] == "SPEAKER_01"
        assert result["segments"][1]["speaker"] == "SPEAKER_00"

    def test_speaker_correction_invalid_old_speaker(self, simple_segments, llm_config):
        """When the LLM references a wrong old_speaker, the correction is skipped."""
        llm_config["llm"]["tasks"]["text_correction"] = False

        backend = MockLLMBackend(
            responses=[
                '[{"line": 0, "old_speaker": "SPEAKER_99", "new_speaker": "SPEAKER_01"}]',
            ]
        )
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        # Speaker should remain unchanged
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert len(result["warnings"]) > 0

    def test_unload_called_even_on_error(self, simple_segments, llm_config):
        """Backend.unload() must be called even if generation raises."""

        class FailingBackend:
            def __init__(self):
                self.loaded = False
                self.unloaded = False

            def load(self):
                self.loaded = True

            def generate(self, prompt, max_tokens=None):
                raise RuntimeError("GPU OOM")

            def unload(self):
                self.unloaded = True

        backend = FailingBackend()
        with pytest.raises(RuntimeError, match="GPU OOM"):
            run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert backend.loaded is True
        assert backend.unloaded is True

    def test_result_structure(self, simple_segments, llm_config):
        llm_config["llm"]["tasks"]["speaker_correction"] = False
        llm_config["llm"]["tasks"]["text_correction"] = False

        backend = MockLLMBackend()
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert "segments" in result
        assert "warnings" in result
        assert isinstance(result["segments"], list)
        assert isinstance(result["warnings"], list)

    def test_pipeline_with_chinese_english(self, chinese_english_segments, llm_config):
        """Full pipeline with code-switched content, only speaker correction enabled."""
        llm_config["llm"]["tasks"]["text_correction"] = False

        # Swap speaker for first segment
        backend = MockLLMBackend(
            responses=[
                '[{"line": 0, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}]',
            ]
        )
        result = run_llm_postprocess(chinese_english_segments, llm_config, backend=backend)

        assert result["warnings"] == []
        assert result["segments"][0]["speaker"] == "SPEAKER_01"
        assert result["segments"][1]["speaker"] == "SPEAKER_01"  # unchanged

    def test_text_correction_json_output(self, llm_config):
        """Verify JSON correction is applied to segment text."""
        llm_config["llm"]["tasks"]["speaker_correction"] = False

        segments = [
            {"speaker": "SPEAKER_00", "text": "today think about it", "start": 0.0, "end": 1.0},
            {
                "speaker": "SPEAKER_01",
                "text": "the 飞机 feedback is good",
                "start": 1.0,
                "end": 2.0,
            },
        ]
        backend = MockLLMBackend(
            responses=[
                '[{"line": 0, "original": "think", "corrected": "sync"}, '
                '{"line": 1, "original": "飞机", "corrected": "VC"}]',
            ]
        )
        result = run_llm_postprocess(segments, llm_config, backend=backend)

        assert result["segments"][0]["text"] == "today sync about it"
        assert result["segments"][1]["text"] == "the VC feedback is good"
        assert result["segments"][0]["start"] == 0.0
        assert result["warnings"] == []

    def test_text_correction_no_errors(self, simple_segments, llm_config):
        """Verify [] response leaves text unchanged."""
        llm_config["llm"]["tasks"]["speaker_correction"] = False

        backend = MockLLMBackend(responses=["[]"])
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert result["segments"][0]["text"] == "Hello world"
        assert result["segments"][1]["text"] == "Hi there"
        assert result["warnings"] == []

    def test_text_correction_malformed_json_is_safe(self, simple_segments, llm_config):
        """Verify bad JSON doesn't crash and leaves text unchanged."""
        llm_config["llm"]["tasks"]["speaker_correction"] = False

        backend = MockLLMBackend(responses=["[{broken json garbage"])
        result = run_llm_postprocess(simple_segments, llm_config, backend=backend)

        assert result["segments"][0]["text"] == "Hello world"
        assert result["segments"][1]["text"] == "Hi there"
        assert result["warnings"] == []
