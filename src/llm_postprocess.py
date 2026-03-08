"""LLM post-processing: speaker correction, text correction.

Uses DiarizationLM-inspired prompting with TPST safety checks to prevent
the LLM from hallucinating or altering transcribed words.
"""

import logging
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger("asr_pipeline")


# ---------------------------------------------------------------------------
# Backend protocol and implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM inference backends."""

    def load(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int | None = None) -> str: ...
    def unload(self) -> None: ...


class LlamaCppBackend:
    """llama-cpp-python backend for GGUF models."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm = None
        self.n_ctx = config.get("llm", {}).get("n_ctx", 8192)

    def load(self) -> None:
        """Load GGUF model via llama-cpp-python."""
        import os
        import sys

        # Ensure CUDA runtime libs from nvidia-* packages are discoverable
        site_packages = next((p for p in sys.path if p.endswith("site-packages")), None)
        if site_packages:
            nvidia_libs = os.path.join(site_packages, "nvidia")
            if os.path.isdir(nvidia_libs):
                lib_dirs = [
                    os.path.join(nvidia_libs, d, "lib")
                    for d in os.listdir(nvidia_libs)
                    if os.path.isdir(os.path.join(nvidia_libs, d, "lib"))
                ]
                ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                for d in lib_dirs:
                    if d not in ld_path:
                        ld_path = d + ":" + ld_path
                os.environ["LD_LIBRARY_PATH"] = ld_path

        from llama_cpp import Llama  # Lazy import

        model_path = self.config["llm"]["model_path"]
        if model_path is None:
            raise ValueError(
                "llm.model_path must be set to a GGUF file path. "
                f"Download from: {self.config['llm'].get('gguf_url', 'N/A')}"
            )

        n_gpu_layers = self.config["llm"].get("n_gpu_layers", -1)
        n_ctx = self.config["llm"].get("n_ctx", 8192)

        logger.info(
            "Loading GGUF model: %s (n_gpu_layers=%s, n_ctx=%d)",
            model_path,
            n_gpu_layers,
            n_ctx,
        )
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            flash_attn=True,
            type_k=8,  # Q8_0 KV cache keys
            type_v=8,  # Q8_0 KV cache values
        )

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate text from prompt.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum number of tokens to generate. Defaults to
                n_ctx // 2 if not specified.
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if max_tokens is None:
            max_tokens = self.n_ctx // 2
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
        )
        return result["choices"][0]["text"]

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            logger.info("LlamaCpp model unloaded.")


class TransformersLLMBackend:
    """transformers + bitsandbytes 4-bit backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.n_ctx = config.get("llm", {}).get("n_ctx", 8192)

    def load(self) -> None:
        """Load model with 4-bit quantization via bitsandbytes."""
        from transformers import (  # Lazy import
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_id = self.config["llm"].get("model_id", "Qwen/Qwen3.5-9B")
        logger.info("Loading transformers model: %s (4-bit)", model_id)

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Tokenize, generate, and decode.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum number of new tokens to generate. Defaults to
                n_ctx // 2 if not specified.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch  # Lazy import

        if max_tokens is None:
            max_tokens = self.n_ctx // 2
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.5,
            )
        # Decode only the newly generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        if self.model is not None or self.tokenizer is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            logger.info("Transformers LLM model unloaded.")


# ---------------------------------------------------------------------------
# Pure functions — all testable without GPU
# ---------------------------------------------------------------------------

_SPEAKER_TAG_RE = re.compile(r"<speaker:(\w+)>")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3.5 ``<think>...</think>`` reasoning blocks from output.

    Handles both closed ``<think>...</think>`` blocks and unclosed ``<think>``
    blocks (when the model runs out of tokens during reasoning).
    """
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def parse_json_corrections(raw: str) -> list[dict]:
    """Parse a JSON array of corrections from LLM output.

    Tolerates preamble text before the JSON array and trailing text after it.
    Returns an empty list on parse failure.
    """
    import json

    raw = raw.strip()
    if not raw:
        return []
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        if raw == "[]":
            return []
        logger.warning("No JSON array found in LLM output: %.200s", raw)
        return []
    try:
        result = json.loads(raw[start : end + 1])
        if not isinstance(result, list):
            logger.warning("JSON output is not an array")
            return []
        return result
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON corrections: %s", e)
        return []


def validate_text_correction(segment_text: str, correction: dict) -> tuple[bool, str]:
    """Validate a single text correction against its source segment.

    Returns:
        ``(is_valid, reason)`` — True if safe to apply, else False with reason.
    """
    original = correction.get("original", "")
    corrected = correction.get("corrected", "")
    if not original or not corrected:
        return False, "Empty original or corrected text"
    if original == corrected:
        return False, "Original and corrected are identical"
    if original not in segment_text:
        return False, f"'{original}' not found in segment"
    if len(corrected) > 3 * len(original) + 10:
        return False, "Correction suspiciously long"
    return True, ""


_MAX_CORRECTIONS_PER_CHUNK = 20


def apply_speaker_corrections(
    segments: list[dict],
    corrections: list[dict],
) -> tuple[list[dict], list[str]]:
    """Apply JSON speaker corrections to segments.

    Each correction must specify the expected old_speaker to prevent
    misapplication if the LLM output is stale or misaligned.

    Args:
        segments: Original transcript segments.
        corrections: List of ``{"line", "old_speaker", "new_speaker"}`` dicts.

    Returns:
        ``(corrected_segments, warnings)``
    """
    warnings: list[str] = []
    result = [dict(s) for s in segments]
    applied = 0
    for corr in corrections:
        line = corr.get("line")
        if not isinstance(line, int) or line < 0 or line >= len(result):
            warnings.append(f"Invalid line {line!r}, skipping speaker correction")
            continue
        old_speaker = corr.get("old_speaker", "")
        new_speaker = corr.get("new_speaker", "")
        if not old_speaker or not new_speaker:
            warnings.append(f"Line {line}: missing speaker field, skipping")
            continue
        if result[line]["speaker"] != old_speaker:
            warnings.append(
                f"Line {line}: expected speaker '{old_speaker}', "
                f"got '{result[line]['speaker']}', skipping"
            )
            continue
        result[line]["speaker"] = new_speaker
        applied += 1
    if applied:
        logger.info("Applied %d speaker correction(s)", applied)
    return result, warnings


def apply_text_corrections(
    segments: list[dict],
    corrections: list[dict],
    *,
    max_corrections: int = _MAX_CORRECTIONS_PER_CHUNK,
) -> tuple[list[dict], list[str]]:
    """Apply JSON text corrections to segments with per-patch validation.

    Each correction is validated individually. Invalid corrections are skipped
    with a warning. If total corrections exceed *max_corrections*, all are
    rejected as likely hallucination.

    Args:
        segments: Original transcript segments.
        corrections: List of ``{"line", "original", "corrected"}`` dicts.
        max_corrections: Hallucination guard — reject all if exceeded.

    Returns:
        ``(corrected_segments, warnings)``
    """
    warnings: list[str] = []
    if len(corrections) > max_corrections:
        warnings.append(
            f"Too many corrections ({len(corrections)} > {max_corrections}), "
            "likely hallucination — skipping all"
        )
        return segments, warnings

    result = [dict(s) for s in segments]
    applied = 0
    for corr in corrections:
        line = corr.get("line")
        if not isinstance(line, int) or line < 0 or line >= len(result):
            warnings.append(f"Invalid line {line!r}, skipping correction")
            continue
        valid, reason = validate_text_correction(result[line]["text"], corr)
        if not valid:
            warnings.append(f"Line {line}: {reason}, skipping")
            continue
        result[line]["text"] = result[line]["text"].replace(corr["original"], corr["corrected"], 1)
        applied += 1
    if applied:
        logger.info("Applied %d text correction(s)", applied)
    return result, warnings


def format_diarization_lm(segments: list[dict]) -> str:
    """Format segments as DiarizationLM text format.

    Each segment becomes a line: ``<speaker:LABEL> text``

    Args:
        segments: List of dicts with ``speaker`` and ``text`` keys.

    Returns:
        Multi-line string in DiarizationLM format.

    Example output::

        <speaker:SPEAKER_00> Hello world
        <speaker:SPEAKER_01> Hi there
    """
    lines: list[str] = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"<speaker:{speaker}> {text}")
    return "\n".join(lines)


def parse_diarization_lm(text: str) -> list[dict]:
    """Parse DiarizationLM format back to a list of segment dicts.

    Args:
        text: Multi-line string where each line starts with ``<speaker:LABEL>``.

    Returns:
        List of ``{"speaker": str, "text": str}`` dicts.
    """
    segments: list[dict] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_TAG_RE.match(line)
        if m:
            speaker = m.group(1)
            rest = line[m.end() :].strip()
            segments.append({"speaker": speaker, "text": rest})
        else:
            # Line without a speaker tag — attach to previous segment or skip.
            if segments:
                segments[-1]["text"] += " " + line
            else:
                segments.append({"speaker": "UNKNOWN", "text": line})
    return segments


def build_speaker_correction_prompt(transcript: str) -> str:
    """Build prompt for speaker label correction with JSON output.

    Instructs the LLM to output a JSON array of speaker label corrections
    instead of reproducing the entire transcript.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full ChatML prompt string ready for LLM generation.
    """
    return (
        "<|im_start|>system\n"
        "You fix speaker labels in diarized transcripts.\n"
        "Output ONLY a JSON array of corrections. If no corrections needed, output [].\n"
        "Each correction: "
        '{"line": N, "old_speaker": "SPEAKER_XX", "new_speaker": "SPEAKER_YY"}\n'
        "where N is the 0-indexed line number.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Example input:\n"
        "<speaker:SPEAKER_00> 你好，我是小明。\n"
        "<speaker:SPEAKER_00> 你好小明，我叫小红。\n"
        "<speaker:SPEAKER_00> 小红你好，今天讨论什么？\n"
        "\n"
        "Example output:\n"
        '[{"line": 1, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}]\n'
        "\n"
        "Now correct speaker labels in this transcript. "
        "Only change labels where speaker attribution is clearly wrong "
        "based on conversational context. If unsure, keep the original label. "
        "If no corrections needed, output [].\n"
        "\n"
        f"{transcript}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n</think>\n"
    )


def build_text_correction_prompt(transcript: str) -> str:
    """Build prompt for text error correction with JSON output.

    Instructs the LLM to output a JSON array of corrections instead of
    reproducing the entire transcript. Each correction identifies the line,
    the original text, and the corrected text.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full ChatML prompt string ready for LLM generation.
    """
    return (
        "<|im_start|>system\n"
        "You fix ASR errors in Chinese-English code-switched transcripts.\n"
        "Output ONLY a JSON array of corrections. If no errors, output [].\n"
        "Each correction: "
        '{"line": N, "original": "wrong text", "corrected": "fixed text"}\n'
        "where N is the 0-indexed line number.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Common ASR error patterns:\n"
        "- English words misheard as Chinese: sync→想/think, demo→带我, VC→飞机\n"
        "- Acronyms garbled: SDK→SCK, PEVC→P2V, SOTA→saota\n"
        "- English phrases heard as Chinese: fund raising→Fun Reason, moat→Mot\n"
        "- Chinese homophones: 拒→剧\n"
        "\n"
        "Example input:\n"
        "<speaker:SPEAKER_00> 今天简单think一下项目进度。\n"
        "<speaker:SPEAKER_01> 好的，我觉得这个飞机给的feedback还行。\n"
        "\n"
        "Example output:\n"
        '[{"line": 0, "original": "think", "corrected": "sync"}, '
        '{"line": 1, "original": "飞机", "corrected": "VC"}]\n'
        "\n"
        "Now fix ASR errors in this transcript. "
        "Only fix clear misrecognitions — do not rephrase, summarize, or merge lines. "
        "If no errors, output [].\n"
        "\n"
        f"{transcript}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n</think>\n"
    )


# ---------------------------------------------------------------------------
# TPST (Transcript-Preserving Speaker Transfer) safety check
# ---------------------------------------------------------------------------

# CJK Unified Ideographs range
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_cjk(char: str) -> bool:
    """Return True if *char* is a CJK unified ideograph."""
    return bool(_CJK_RE.match(char))


def _extract_words(segments: list[dict]) -> list[str]:
    """Extract normalised word/character tokens from segments.

    1. Strip speaker tags.
    2. Strip punctuation (keep only word characters and CJK).
    3. For CJK characters: each character becomes its own token.
    4. For Latin text: whitespace-delimited word tokens, lowercased.

    Returns:
        Flat list of tokens suitable for equality comparison.
    """
    # Concatenate all segment texts
    texts: list[str] = []
    for seg in segments:
        text = seg.get("text", "")
        texts.append(text)
    raw = " ".join(texts)

    # Strip speaker tags that might remain
    raw = re.sub(r"<speaker:\w+>\s*", "", raw)

    tokens: list[str] = []
    # Split into chunks: CJK characters are individual tokens,
    # everything else is whitespace-split words.
    # We iterate character by character, grouping non-CJK runs.
    buf: list[str] = []

    for ch in raw:
        if _is_cjk(ch):
            # Flush any accumulated Latin buffer
            if buf:
                word = "".join(buf).strip()
                if word:
                    # Remove non-word chars (punctuation) then split
                    cleaned = re.sub(r"[^\w]", " ", word, flags=re.UNICODE)
                    for w in cleaned.split():
                        if w:
                            tokens.append(w.lower())
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)

    # Flush remaining buffer
    if buf:
        word = "".join(buf).strip()
        if word:
            cleaned = re.sub(r"[^\w]", " ", word, flags=re.UNICODE)
            for w in cleaned.split():
                if w:
                    tokens.append(w.lower())

    return tokens


def tpst_check(original: list[dict], corrected: list[dict]) -> tuple[list[dict], list[str]]:
    """TPST (Transcript-Preserving Speaker Transfer) safety check.

    Verifies the LLM only changed speaker labels and punctuation, never
    altered the actual words.

    Algorithm:
        - Extract word/character tokens from both original and corrected
          (stripping speaker tags and punctuation).
        - CJK characters: character-level comparison.
        - Latin text: word-level comparison (case-insensitive).
        - If tokens differ: return ``(original_segments, [warnings])``.
        - If tokens match: return ``(corrected_segments, [])``.

    Args:
        original: Segments before LLM processing.
        corrected: Segments after LLM processing.

    Returns:
        ``(segments_to_use, warnings)`` — corrected segments if safe,
        original segments with warnings otherwise.
    """
    if not original and not corrected:
        return [], []

    orig_tokens = _extract_words(original)
    corr_tokens = _extract_words(corrected)

    if orig_tokens == corr_tokens:
        return corrected, []

    # Build a human-readable diff summary
    warnings: list[str] = []

    if len(orig_tokens) != len(corr_tokens):
        warnings.append(
            f"TPST: Token count mismatch — original has {len(orig_tokens)} "
            f"tokens, corrected has {len(corr_tokens)} tokens."
        )
    else:
        diffs: list[str] = []
        for i, (o, c) in enumerate(zip(orig_tokens, corr_tokens, strict=False)):
            if o != c:
                diffs.append(f"  pos {i}: '{o}' -> '{c}'")
                if len(diffs) >= 5:
                    diffs.append("  ... (truncated)")
                    break
        if diffs:
            detail = "\n".join(diffs)
            warnings.append(f"TPST: Words were modified:\n{detail}")

    warnings.append(
        "TPST: Falling back to original segments because the LLM altered transcribed words."
    )

    for w in warnings:
        logger.warning(w)

    return original, warnings


# ---------------------------------------------------------------------------
# Chunking for long transcripts
# ---------------------------------------------------------------------------


def _chunk_segments(
    segments: list[dict],
    max_segments_per_chunk: int,
) -> list[list[dict]]:
    """Split segments into chunks, preferring speaker-turn boundaries.

    Avoids splitting in the middle of a speaker turn (consecutive segments
    with the same speaker).

    Args:
        segments: All transcript segments.
        max_segments_per_chunk: Target max segments per chunk.

    Returns:
        List of segment chunks. All original segments appear exactly once.
    """
    if not segments:
        return []
    if len(segments) <= max_segments_per_chunk:
        return [segments]

    # Find speaker-turn boundary indices
    boundaries: list[int] = [0]
    for i in range(1, len(segments)):
        if segments[i].get("speaker") != segments[i - 1].get("speaker"):
            boundaries.append(i)

    chunks: list[list[dict]] = []
    chunk_start = 0
    while chunk_start < len(segments):
        chunk_end = min(chunk_start + max_segments_per_chunk, len(segments))
        if chunk_end < len(segments):
            # Find nearest speaker-turn boundary at or before chunk_end
            best = chunk_start + 1  # minimum 1 segment
            for b in boundaries:
                if chunk_start < b <= chunk_end:
                    best = b
            chunk_end = best
        chunks.append(segments[chunk_start:chunk_end])
        chunk_start = chunk_end
    return chunks


def _process_json_speaker_output(
    original_chunk: list[dict], raw_output: str
) -> tuple[list[dict], list[str]]:
    """Process JSON speaker corrections output."""
    corrections = parse_json_corrections(raw_output)
    return apply_speaker_corrections(original_chunk, corrections)


def _process_json_text_output(
    original_chunk: list[dict], raw_output: str
) -> tuple[list[dict], list[str]]:
    """Process JSON text corrections output."""
    corrections = parse_json_corrections(raw_output)
    return apply_text_corrections(original_chunk, corrections)


def _apply_correction_chunked(
    segments: list[dict],
    prompt_builder,
    backend,
    n_ctx: int,
    *,
    output_processor,
    max_output_tokens: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Apply LLM correction in chunks that fit the context window.

    Estimates tokens per segment, splits into chunks, processes each,
    and reassembles.

    Args:
        segments: All transcript segments.
        prompt_builder: Function that builds prompt from formatted text.
        backend: LLM backend with .generate() method.
        n_ctx: Context window size in tokens.
        output_processor: Callback ``(original_chunk, raw_output) -> (segments, warnings)``
            that parses and validates the LLM output for a single chunk.
        max_output_tokens: If set, pass to ``backend.generate()`` and use as
            the output budget for chunk sizing.

    Returns:
        (corrected_segments, warnings)
    """
    # Build a trial prompt to measure actual size, then estimate tokens.
    # CJK-heavy text tokenizes at ~2 chars/token in Qwen; mixed text ~2.5.
    # We use 2 as a conservative estimate to avoid under-chunking.
    chars_per_token = 2
    formatted = format_diarization_lm(segments)
    trial_prompt = prompt_builder(formatted)
    estimated_total_input = len(trial_prompt) / chars_per_token
    estimated_transcript_tokens = len(formatted) / chars_per_token
    output_budget = max_output_tokens if max_output_tokens else n_ctx // 2

    def _process_chunk(original_chunk):
        """Generate, parse, validate, and merge metadata for one chunk."""
        fmt = format_diarization_lm(original_chunk)
        prompt = prompt_builder(fmt)
        raw_output = _strip_think_tags(backend.generate(prompt, max_tokens=max_output_tokens))
        return output_processor(original_chunk, raw_output)

    if estimated_total_input + output_budget <= n_ctx:
        # Fits in one shot
        return _process_chunk(segments)

    # Need to chunk
    overhead_tokens = estimated_total_input - estimated_transcript_tokens
    available_for_transcript = n_ctx - overhead_tokens - output_budget
    tokens_per_segment = max(1, estimated_transcript_tokens / max(1, len(segments)))
    max_segments = max(1, int(available_for_transcript / tokens_per_segment))
    chunks = _chunk_segments(segments, max_segments)
    logger.info(
        "Transcript too long (%d segments, ~%d tokens, ~%d available), "
        "splitting into %d chunks of ~%d segments",
        len(segments),
        int(estimated_transcript_tokens),
        int(available_for_transcript),
        len(chunks),
        max_segments,
    )

    all_corrected = []
    all_warnings = []
    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d (%d segments)", i + 1, len(chunks), len(chunk))
        safe_segments, warnings = _process_chunk(chunk)
        all_corrected.extend(safe_segments)
        all_warnings.extend(warnings)

    return all_corrected, all_warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_llm_postprocess(
    segments: list[dict],
    config: dict,
    backend: LLMBackend | None = None,
) -> dict:
    """Run LLM post-processing pipeline.

    Steps:
        1. Create backend (LlamaCpp or Transformers) if not provided.
        2. ``backend.load()``
        3. **Task 1** — Speaker correction (if enabled):
           ``format -> prompt -> generate -> parse -> tpst_check``
        4. **Task 2** — Text correction (if enabled):
           ``format -> prompt -> generate -> parse``
        5. ``backend.unload()``
        6. Return result dict.

    Args:
        segments: List of transcript segment dicts (``speaker``, ``text``, etc.).
        config: Full pipeline configuration dictionary.
        backend: Optional pre-created backend (useful for testing / injection).

    Returns:
        Dict with keys ``segments`` and ``warnings``.
    """
    tasks = config.get("llm", {}).get("tasks", {})
    all_warnings: list[str] = []
    current_segments = segments

    # 1. Resolve backend
    if backend is None:
        backend_name = config.get("llm", {}).get("backend", "llama-cpp")
        if backend_name == "llama-cpp":
            backend = LlamaCppBackend(config)
        elif backend_name == "transformers":
            backend = TransformersLLMBackend(config)
        else:
            raise ValueError(f"Unknown LLM backend: {backend_name}")

    n_ctx = config.get("llm", {}).get("n_ctx", 8192)

    # 2. Load
    backend.load()

    try:
        # 3. Task 1: Speaker correction
        if tasks.get("speaker_correction", False):
            logger.info("Running LLM task: speaker correction")
            current_segments, warnings = _apply_correction_chunked(
                current_segments,
                build_speaker_correction_prompt,
                backend,
                n_ctx,
                output_processor=_process_json_speaker_output,
                max_output_tokens=1024,
            )
            all_warnings.extend(warnings)

        # 4. Task 2: Text correction (JSON error-pair output)
        if tasks.get("text_correction", False):
            logger.info("Running LLM task: text correction")
            current_segments, warnings = _apply_correction_chunked(
                current_segments,
                build_text_correction_prompt,
                backend,
                n_ctx,
                output_processor=_process_json_text_output,
                max_output_tokens=1024,
            )
            all_warnings.extend(warnings)

    finally:
        backend.unload()

    return {
        "segments": current_segments,
        "warnings": all_warnings,
    }
