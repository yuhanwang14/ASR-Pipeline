# ASR Evaluation Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `scripts/evaluate.py` that computes CER, WER, and cpWER against a Gemini reference transcript, with a printed scorecard.

**Architecture:** Pure metric functions in `src/eval_metrics.py` (testable without GPU), CLI orchestration in `scripts/evaluate.py`. Uses `jiwer` for edit-distance computation.

**Tech Stack:** Python, jiwer, argparse

---

### Task 1: Add jiwer dependency

**Files:**
- Modify: `pyproject.toml:44-49` (dev dependencies)

**Step 1: Add jiwer to dev dependencies**

In `[project.optional-dependencies]` under `dev`, add `"jiwer>=3.0"`:

```toml
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff>=0.1.0",
    "pytest-mock",
    "jiwer>=3.0",
]
```

**Step 2: Install**

Run: `uv sync --extra dev`
Expected: jiwer installs successfully.

**Step 3: Verify import**

Run: `uv run python -c "import jiwer; print(jiwer.__version__)"`
Expected: Prints version number.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add jiwer for ASR evaluation metrics"
```

---

### Task 2: Reference parser and pipeline output loader

**Files:**
- Create: `src/eval_metrics.py`
- Create: `tests/test_eval_metrics.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: ImportError — `src.eval_metrics` does not exist.

**Step 3: Implement**

```python
"""ASR evaluation metrics: CER, WER, cpWER against reference transcripts."""

import json
import re
from pathlib import Path

_SPEAKER_LINE_RE = re.compile(r"^\*\*Speaker (\d+):\*\*\s*(.+)$")
_TIMESTAMP_RE = re.compile(r"^\(\d+:\d+\)$")


def parse_reference_transcript(markdown: str) -> list[dict]:
    """Parse Gemini-style markdown transcript into segments.

    Expected format: ``**Speaker N:** text`` lines separated by blank lines.
    Skips blank lines, ``**Transcript**`` header, and ``(M:SS)`` timestamps.
    """
    segments: list[dict] = []
    for line in markdown.strip().splitlines():
        line = line.strip()
        if not line or _TIMESTAMP_RE.match(line):
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if m:
            speaker = f"Speaker {m.group(1)}"
            text = m.group(2).strip()
            if text:
                segments.append({"speaker": speaker, "text": text})
    return segments


def load_pipeline_output(path: str | Path) -> list[dict]:
    """Load pipeline JSON output and return filtered segments.

    Filters out empty-text segments and segments with leaked ``<think>`` blocks.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    segments = data.get("segments", [])
    return [
        s for s in segments
        if s.get("text", "").strip() and not s.get("text", "").startswith("<think>")
    ]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add reference parser and pipeline output loader"
```

---

### Task 3: Text metrics — CER and mixed WER

**Files:**
- Modify: `src/eval_metrics.py`
- Modify: `tests/test_eval_metrics.py`

**Step 1: Write failing tests**

```python
from src.eval_metrics import compute_cer, compute_wer, normalize_for_eval


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py::TestNormalizeForEval tests/test_eval_metrics.py::TestComputeCer tests/test_eval_metrics.py::TestComputeWer -v`
Expected: ImportError — functions not defined.

**Step 3: Implement**

Add to `src/eval_metrics.py`:

```python
import jiwer

_PUNCTUATION_RE = re.compile(
    r"[，。！？、；：""''（）《》【】…·\.\,\!\?\;\:\"\'\(\)\[\]\-\—\.\.\.]+"
)


def normalize_for_eval(text: str) -> str:
    """Normalize text for evaluation: strip punctuation, lowercase, collapse whitespace."""
    text = _PUNCTUATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _tokenize_mixed(text: str) -> list[str]:
    """Tokenize mixed Chinese-English text.

    Each CJK character becomes one token. English words (whitespace-delimited)
    become one token each. This matches the standard "mixed error rate" used
    in Chinese-English ASR evaluation.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                word = "".join(buf).strip()
                if word:
                    tokens.extend(word.split())
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        word = "".join(buf).strip()
        if word:
            tokens.extend(word.split())
    return tokens


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    ref = normalize_for_eval(reference)
    hyp = normalize_for_eval(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute mixed Word Error Rate (CJK chars = tokens, English words = tokens)."""
    ref_tokens = _tokenize_mixed(normalize_for_eval(reference))
    hyp_tokens = _tokenize_mixed(normalize_for_eval(hypothesis))
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add CER and mixed-WER computation"
```

---

### Task 4: cpWER with optimal speaker mapping

**Files:**
- Modify: `src/eval_metrics.py`
- Modify: `tests/test_eval_metrics.py`

**Step 1: Write failing tests**

```python
from src.eval_metrics import compute_cpwer


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py::TestComputeCpwer -v`
Expected: ImportError.

**Step 3: Implement**

Add to `src/eval_metrics.py`:

```python
from itertools import permutations


def compute_cpwer(
    ref_segments: list[dict],
    hyp_segments: list[dict],
) -> dict:
    """Compute concatenated minimum-permutation WER (cpWER).

    Finds the optimal mapping of hypothesis speakers to reference speakers
    by trying all permutations and picking the one with minimum WER.

    Returns dict with keys: ``cpwer``, ``mapping``, ``per_speaker``.
    """
    if not ref_segments and not hyp_segments:
        return {"cpwer": 0.0, "mapping": {}, "per_speaker": {}}

    ref_speakers = sorted(set(s["speaker"] for s in ref_segments))
    hyp_speakers = sorted(set(s["speaker"] for s in hyp_segments))

    # Concatenate normalized text per speaker
    ref_by_speaker = {
        sp: " ".join(normalize_for_eval(s["text"]) for s in ref_segments if s["speaker"] == sp)
        for sp in ref_speakers
    }
    hyp_by_speaker = {
        sp: " ".join(normalize_for_eval(s["text"]) for s in hyp_segments if s["speaker"] == sp)
        for sp in hyp_speakers
    }

    # Pad to equal length with dummy speakers
    n = max(len(ref_speakers), len(hyp_speakers))
    padded_ref = ref_speakers + [f"_UNMAPPED_REF_{i}" for i in range(n - len(ref_speakers))]
    padded_hyp = hyp_speakers + [f"_UNMAPPED_HYP_{i}" for i in range(n - len(hyp_speakers))]

    best_errors = float("inf")
    best_ref_len = 1
    best_mapping: dict[str, str] = {}

    for perm in permutations(range(n)):
        mapping = {padded_hyp[i]: padded_ref[perm[i]] for i in range(n)}
        total_errors = 0
        total_ref_len = 0

        for hyp_sp, ref_sp in mapping.items():
            ref_text = ref_by_speaker.get(ref_sp, "")
            hyp_text = hyp_by_speaker.get(hyp_sp, "")
            ref_tokens = _tokenize_mixed(ref_text)
            hyp_tokens = _tokenize_mixed(hyp_text)

            if ref_tokens:
                measures = jiwer.compute_measures(
                    " ".join(ref_tokens), " ".join(hyp_tokens) if hyp_tokens else ""
                )
                total_errors += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total_ref_len += len(ref_tokens)
            elif hyp_tokens:
                total_errors += len(hyp_tokens)

        if total_errors < best_errors:
            best_errors = total_errors
            best_ref_len = max(total_ref_len, 1)
            best_mapping = {h: r for h, r in mapping.items() if not h.startswith("_UNMAPPED")}

    cpwer = best_errors / best_ref_len

    # Per-speaker breakdown
    per_speaker: dict[str, dict] = {}
    for hyp_sp, ref_sp in best_mapping.items():
        ref_text = ref_by_speaker.get(ref_sp, "")
        hyp_text = hyp_by_speaker.get(hyp_sp, "")
        if ref_text:
            per_speaker[ref_sp] = {
                "hyp_speaker": hyp_sp,
                "cer": compute_cer(ref_text, hyp_text),
                "wer": compute_wer(ref_text, hyp_text),
            }

    return {"cpwer": cpwer, "mapping": best_mapping, "per_speaker": per_speaker}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add cpWER with optimal speaker mapping"
```

---

### Task 5: CLI script and manual test

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: Write the CLI script**

```python
#!/usr/bin/env python3
"""Evaluate ASR pipeline output against a reference transcript.

Usage:
    # Evaluate cached output (fast):
    uv run scripts/evaluate.py --output "output/zoom-mar-7/Zoom - Mar 7.json" \
        --reference "tests/fixtures/Zoom - Mar 7.md"

    # Run pipeline + evaluate:
    uv run scripts/evaluate.py "tests/fixtures/Zoom - Mar 7.m4a" \
        --reference "tests/fixtures/Zoom - Mar 7.md"
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR pipeline output")
    parser.add_argument("audio_file", nargs="?", help="Audio file (runs pipeline first)")
    parser.add_argument("--output", help="Pre-computed pipeline JSON output (skip pipeline)")
    parser.add_argument("--reference", required=True, help="Reference transcript (.md)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config file")
    parser.add_argument("--output-dir", default=None, help="Pipeline output directory")

    args = parser.parse_args()

    if not args.audio_file and not args.output:
        parser.error("Provide either an audio file or --output path")

    from src.eval_metrics import (
        compute_cer,
        compute_cpwer,
        compute_wer,
        load_pipeline_output,
        normalize_for_eval,
        parse_reference_transcript,
    )

    # Load reference
    ref_text = Path(args.reference).read_text(encoding="utf-8")
    ref_segments = parse_reference_transcript(ref_text)
    if not ref_segments:
        print("Error: no segments found in reference transcript", file=sys.stderr)
        sys.exit(1)

    # Load or generate hypothesis
    if args.output:
        hyp_segments = load_pipeline_output(args.output)
        src_label = args.output
    else:
        from src.config import load_config
        from src.logging_config import setup_logging
        from src.pipeline import run_pipeline

        setup_logging(verbose=True)
        config = load_config(args.config)
        if args.output_dir:
            config["output"]["output_dir"] = args.output_dir
        result = run_pipeline(args.audio_file, config=config)
        hyp_segments = [
            s for s in result["segments"]
            if s.get("text", "").strip() and not s.get("text", "").startswith("<think>")
        ]
        src_label = args.audio_file

    if not hyp_segments:
        print("Error: no segments found in pipeline output", file=sys.stderr)
        sys.exit(1)

    # Text-level metrics (speaker-agnostic)
    ref_full = " ".join(normalize_for_eval(s["text"]) for s in ref_segments)
    hyp_full = " ".join(normalize_for_eval(s["text"]) for s in hyp_segments)
    cer = compute_cer(ref_full, hyp_full)
    wer = compute_wer(ref_full, hyp_full)

    # Speaker-aware metrics
    cpwer_result = compute_cpwer(ref_segments, hyp_segments)

    # Print scorecard
    ref_speakers = sorted(set(s["speaker"] for s in ref_segments))
    hyp_speakers = sorted(set(s["speaker"] for s in hyp_segments))

    print("=== ASR Pipeline Evaluation ===")
    print(f"Reference:  {args.reference} ({len(ref_speakers)} speakers, {len(ref_segments)} segments)")
    print(f"Hypothesis: {src_label} ({len(hyp_speakers)} speakers, {len(hyp_segments)} segments)")
    print()
    print("Text Quality (speaker-agnostic):")
    print(f"  CER: {cer:6.1%}")
    print(f"  WER: {wer:6.1%}")
    print()
    print("Speaker Diarization (cpWER):")
    mapping = cpwer_result["mapping"]
    mapping_str = ", ".join(f"{h}\u2192{r}" for h, r in sorted(mapping.items()))
    print(f"  Optimal mapping: {mapping_str}")
    print(f"  cpWER: {cpwer_result['cpwer']:6.1%}")
    print()
    if cpwer_result["per_speaker"]:
        print("  Per-speaker:")
        for ref_sp, metrics in sorted(cpwer_result["per_speaker"].items()):
            hyp_sp = metrics["hyp_speaker"]
            print(
                f"    {ref_sp} (\u2192{hyp_sp}):"
                f"  CER={metrics['cer']:.1%}"
                f"  WER={metrics['wer']:.1%}"
            )


if __name__ == "__main__":
    main()
```

**Step 2: Test with cached output**

Run: `uv run scripts/evaluate.py --output "output/zoom-mar-7/Zoom - Mar 7.json" --reference "tests/fixtures/Zoom - Mar 7.md"`

Expected: Scorecard prints with CER, WER, cpWER, and per-speaker breakdown. No crashes.

**Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add ASR evaluation script with CER/WER/cpWER scorecard"
```

---

### Task 6: Lint and final test

**Step 1: Run ruff**

Run: `uv run ruff check src/eval_metrics.py scripts/evaluate.py tests/test_eval_metrics.py --fix`
Run: `uv run ruff format src/eval_metrics.py scripts/evaluate.py tests/test_eval_metrics.py`

**Step 2: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_gpu_integration.py -v`
Expected: All tests pass including new `test_eval_metrics.py`.

**Step 3: Commit if fixes were needed**

```bash
git add -u
git commit -m "style: apply ruff formatting to eval metrics"
```
