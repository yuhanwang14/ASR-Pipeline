"""LLM post-processing: speaker correction, text correction, summarization.

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
    def generate(self, prompt: str) -> str: ...
    def unload(self) -> None: ...


class LlamaCppBackend:
    """llama-cpp-python backend for GGUF models."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm = None

    def load(self) -> None:
        """Load GGUF model via llama-cpp-python."""
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
        )

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        result = self.llm(prompt, max_tokens=2048)
        return result["choices"][0]["text"]

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        from src.gpu_utils import unload_model  # Lazy import

        if self.llm is not None:
            llm_ref = self.llm
            self.llm = None
            unload_model(llm_ref)
            logger.info("LlamaCpp model unloaded.")


class TransformersLLMBackend:
    """transformers + bitsandbytes 4-bit backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None

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

    def generate(self, prompt: str) -> str:
        """Tokenize, generate, and decode."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch  # Lazy import

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )
        # Decode only the newly generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        from src.gpu_utils import unload_model  # Lazy import

        refs = []
        if self.model is not None:
            refs.append(self.model)
            self.model = None
        if self.tokenizer is not None:
            refs.append(self.tokenizer)
            self.tokenizer = None
        if refs:
            unload_model(*refs)
            logger.info("Transformers LLM model unloaded.")


# ---------------------------------------------------------------------------
# Pure functions — all testable without GPU
# ---------------------------------------------------------------------------

_SPEAKER_TAG_RE = re.compile(r"<speaker:(\w+)>")


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
    """Build prompt for speaker label correction task.

    Instructs the LLM to only change ``<speaker:XX>`` tags and never touch the
    transcribed words.  Template follows the DiarizationLM approach.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full prompt string ready for LLM generation.
    """
    return (
        "You are a transcript editor. Below is a speaker-diarized transcript that may have "
        "speaker attribution errors. Based on conversational context and semantic coherence, "
        "correct the speaker labels ONLY. Do NOT modify any words in the transcript.\n"
        "\n"
        "Rules:\n"
        "1. Only change <speaker:XX> tags\n"
        "2. Never add, remove, or modify any spoken words\n"
        "3. Use conversational flow to determine correct speaker assignments\n"
        "4. If unsure, keep the original label\n"
        "\n"
        "Transcript:\n"
        f"{transcript}\n"
        "\n"
        "Output the corrected transcript in the same format:\n"
    )


def build_text_correction_prompt(transcript: str) -> str:
    """Build prompt for text error correction task.

    Instructs the LLM to fix ASR errors, homophones, and punctuation while
    preserving code-switching style. Speaker labels must remain untouched.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full prompt string ready for LLM generation.
    """
    return (
        "You are a transcript proofreader for Chinese-English code-switched speech. "
        "Fix obvious ASR errors (homophones, misheard words) while preserving the "
        "original meaning. Fix punctuation. Do NOT change speaker labels.\n"
        "\n"
        "Rules:\n"
        "1. Fix clear errors only — do not rephrase or paraphrase\n"
        "2. Preserve code-switching style (don't translate Chinese to English or vice versa)\n"
        "3. Fix punctuation and sentence boundaries\n"
        "4. Keep all speaker labels unchanged\n"
        "\n"
        "Transcript:\n"
        f"{transcript}\n"
        "\n"
        "Output the corrected transcript:\n"
    )


def build_summary_prompt(transcript: str) -> str:
    """Build prompt for summary generation.

    Asks for key points, decisions, and action items in the same language mix
    as the original transcript.

    Args:
        transcript: Transcript in DiarizationLM text format (or plain text).

    Returns:
        Full prompt string ready for LLM generation.
    """
    return (
        "Summarize the following meeting transcript. Include:\n"
        "1. Key discussion points\n"
        "2. Decisions made\n"
        "3. Action items with assigned speakers\n"
        "Output in the same language mix as the original transcript.\n"
        "\n"
        "Transcript:\n"
        f"{transcript}\n"
        "\n"
        "Summary:\n"
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
           ``format -> prompt -> generate -> parse -> tpst_check``
        5. **Task 3** — Summary (if enabled):
           ``prompt -> generate``
        6. ``backend.unload()``
        7. Return result dict.

    Args:
        segments: List of transcript segment dicts (``speaker``, ``text``, etc.).
        config: Full pipeline configuration dictionary.
        backend: Optional pre-created backend (useful for testing / injection).

    Returns:
        Dict with keys ``segments``, ``summary``, and ``warnings``.
    """
    tasks = config.get("llm", {}).get("tasks", {})
    all_warnings: list[str] = []
    current_segments = segments
    summary: str | None = None

    # 1. Resolve backend
    if backend is None:
        backend_name = config.get("llm", {}).get("backend", "llama-cpp")
        if backend_name == "llama-cpp":
            backend = LlamaCppBackend(config)
        elif backend_name == "transformers":
            backend = TransformersLLMBackend(config)
        else:
            raise ValueError(f"Unknown LLM backend: {backend_name}")

    # 2. Load
    backend.load()

    try:
        # 3. Task 1: Speaker correction
        if tasks.get("speaker_correction", False):
            logger.info("Running LLM task: speaker correction")
            formatted = format_diarization_lm(current_segments)
            prompt = build_speaker_correction_prompt(formatted)
            raw_output = backend.generate(prompt)
            corrected = parse_diarization_lm(raw_output)
            current_segments, warnings = tpst_check(current_segments, corrected)
            all_warnings.extend(warnings)

        # 4. Task 2: Text correction
        if tasks.get("text_correction", False):
            logger.info("Running LLM task: text correction")
            formatted = format_diarization_lm(current_segments)
            prompt = build_text_correction_prompt(formatted)
            raw_output = backend.generate(prompt)
            corrected = parse_diarization_lm(raw_output)
            current_segments, warnings = tpst_check(current_segments, corrected)
            all_warnings.extend(warnings)

        # 5. Task 3: Summary
        if tasks.get("summarization", False):
            logger.info("Running LLM task: summarization")
            formatted = format_diarization_lm(current_segments)
            prompt = build_summary_prompt(formatted)
            summary = backend.generate(prompt)

    finally:
        # 6. Always unload
        backend.unload()

    # 7. Return result
    return {
        "segments": current_segments,
        "summary": summary,
        "warnings": all_warnings,
    }
