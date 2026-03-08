"""ASR evaluation metrics: CER, WER, cpWER against reference transcripts."""

import json
import re
from itertools import permutations
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
                measures = jiwer.process_words(
                    " ".join(ref_tokens), " ".join(hyp_tokens) if hyp_tokens else ""
                )
                total_errors += measures.substitutions + measures.deletions + measures.insertions
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
