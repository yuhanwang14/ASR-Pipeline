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
    Uses Qwen ChatML format for reliable instruction following.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full prompt string ready for LLM generation.
    """
    n_lines = len(transcript.strip().splitlines())
    return (
        "<|im_start|>system\n"
        "You correct speaker labels in diarized transcripts. "
        "You MUST output every line verbatim, only changing <speaker:XX> tags.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Example input:\n"
        "<speaker:SPEAKER_00> 你好，我是小明。\n"
        "<speaker:SPEAKER_00> 你好小明，我叫小红。\n"
        "<speaker:SPEAKER_00> 小红你好，今天讨论什么？\n"
        "\n"
        "Example output:\n"
        "<speaker:SPEAKER_00> 你好，我是小明。\n"
        "<speaker:SPEAKER_01> 你好小明，我叫小红。\n"
        "<speaker:SPEAKER_00> 小红你好，今天讨论什么？\n"
        "\n"
        f"Now correct this transcript. Output EXACTLY {n_lines} lines. "
        "Copy every word verbatim — only change <speaker:XX> tags where the speaker "
        "attribution is clearly wrong based on conversational context. "
        "If unsure, keep the original label.\n"
        "\n"
        f"{transcript}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n</think>\n"
    )


def build_text_correction_prompt(transcript: str) -> str:
    """Build prompt for text error correction task.

    Instructs the LLM to fix ASR errors, homophones, and punctuation while
    preserving code-switching style. Speaker labels must remain untouched.
    Uses Qwen ChatML format for reliable instruction following.

    Args:
        transcript: Transcript in DiarizationLM text format.

    Returns:
        Full prompt string ready for LLM generation.
    """
    n_lines = len(transcript.strip().splitlines())
    return (
        "<|im_start|>system\n"
        "You proofread Chinese-English code-switched ASR transcripts. "
        "Fix recognition errors while preserving every line and every speaker label exactly.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Common ASR errors in Chinese-English code-switching:\n"
        "- English words misheard as Chinese: sync→想/think, demo→带我, VC→飞机\n"
        "- Acronyms garbled: SDK→SCK, PEVC→P2V, SOTA→saota\n"
        "- English phrases heard as Chinese: fund raising→Fun Reason, moat→Mot\n"
        "- Chinese homophones: 拒→剧, funding→founding\n"
        "\n"
        "Example input:\n"
        "<speaker:SPEAKER_00> 今天简单think一下项目进度。\n"
        "<speaker:SPEAKER_01> 好的，我觉得这个飞机给的feedback还行。\n"
        "\n"
        "Example output:\n"
        "<speaker:SPEAKER_00> 今天简单sync一下项目进度。\n"
        "<speaker:SPEAKER_01> 好的，我觉得这个VC给的feedback还行。\n"
        "\n"
        f"Now fix ASR errors in this transcript. Output EXACTLY {n_lines} lines. "
        "Keep all <speaker:XX> tags and line structure unchanged. "
        "Only fix clear misrecognitions — do not rephrase, summarize, or merge lines.\n"
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


def _chunk_segments(segments: list[dict], max_segments_per_chunk: int) -> list[list[dict]]:
    """Split segments into chunks that fit within context window.

    Args:
        segments: All transcript segments.
        max_segments_per_chunk: Max segments per chunk.

    Returns:
        List of segment chunks.
    """
    if not segments:
        return []
    chunks = []
    for i in range(0, len(segments), max_segments_per_chunk):
        chunks.append(segments[i : i + max_segments_per_chunk])
    return chunks


def _apply_correction_chunked(
    segments: list[dict],
    prompt_builder,
    backend,
    n_ctx: int,
    *,
    use_tpst: bool = True,
) -> tuple[list[dict], list[str]]:
    """Apply LLM correction in chunks that fit the context window.

    Estimates tokens per segment, splits into chunks, processes each,
    and reassembles.

    Args:
        segments: All transcript segments.
        prompt_builder: Function that builds prompt from formatted text.
        backend: LLM backend with .generate() method.
        n_ctx: Context window size in tokens.
        use_tpst: If True, run TPST safety check on each chunk (appropriate
            for speaker correction where words must not change). If False,
            accept the LLM output directly (appropriate for text correction
            where fixing misrecognised words is the goal).

    Returns:
        (corrected_segments, warnings)
    """
    # Build a trial prompt to measure actual size, then estimate tokens.
    # CJK-heavy text tokenizes at ~2 chars/token in Qwen; mixed text ~2.5.
    # We use 2 as a conservative estimate to avoid under-chunking.
    formatted = format_diarization_lm(segments)
    trial_prompt = prompt_builder(formatted)
    chars_per_token = 2
    estimated_prompt_tokens = len(trial_prompt) / chars_per_token
    estimated_transcript_tokens = len(formatted) / chars_per_token
    # Output must fit: prompt + output <= n_ctx
    available_tokens = n_ctx - estimated_prompt_tokens

    def _process_chunk(original_chunk):
        """Generate, parse, validate, and merge metadata for one chunk."""
        fmt = format_diarization_lm(original_chunk)
        prompt = prompt_builder(fmt)
        raw_output = _strip_think_tags(backend.generate(prompt))
        corrected = parse_diarization_lm(raw_output)

        if use_tpst:
            corrected, warnings = tpst_check(original_chunk, corrected)
        else:
            warnings = []

        # Merge corrected speaker/text back into originals to preserve
        # timestamps and other metadata (start, end, etc.).
        merged = []
        for j, orig in enumerate(original_chunk):
            entry = dict(orig)  # shallow copy — keeps start, end, etc.
            if j < len(corrected):
                entry["speaker"] = corrected[j].get("speaker", orig.get("speaker"))
                entry["text"] = corrected[j].get("text", orig.get("text", ""))
            merged.append(entry)
        return merged, warnings

    if estimated_transcript_tokens <= available_tokens:
        # Fits in one shot
        return _process_chunk(segments)

    # Need to chunk
    tokens_per_segment = estimated_transcript_tokens / len(segments)
    max_segments = max(1, int(available_tokens / tokens_per_segment))
    chunks = _chunk_segments(segments, max_segments)
    logger.info(
        "Transcript too long (%d segments, ~%d tokens, ~%d available), "
        "splitting into %d chunks of ~%d segments",
        len(segments),
        int(estimated_transcript_tokens),
        int(available_tokens),
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
            )
            all_warnings.extend(warnings)

        # 4. Task 2: Text correction (no TPST — fixing words is the goal)
        if tasks.get("text_correction", False):
            logger.info("Running LLM task: text correction")
            current_segments, warnings = _apply_correction_chunked(
                current_segments,
                build_text_correction_prompt,
                backend,
                n_ctx,
                use_tpst=False,
            )
            all_warnings.extend(warnings)

    finally:
        backend.unload()

    return {
        "segments": current_segments,
        "warnings": all_warnings,
    }
