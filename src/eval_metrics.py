"""ASR evaluation metrics: CER, WER, cpWER against reference transcripts."""

import json
import re
from pathlib import Path

import jiwer

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


_PUNCTUATION_RE = re.compile(
    r"[，。！？、；：\u201c\u201d\u2018\u2019（）《》【】…·.,!?;:\"'()\[\]\-—]+"
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
