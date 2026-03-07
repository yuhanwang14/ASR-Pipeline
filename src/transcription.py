"""Stage 2: Speech-to-text transcription using Qwen3-ASR-1.7B."""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger("asr_pipeline")


@runtime_checkable
class ASRBackend(Protocol):
    """Protocol for ASR backend implementations."""

    def load(self) -> None: ...
    def transcribe(self, audio_path: str) -> str: ...
    def unload(self) -> None: ...


class VLLMBackend:
    """Qwen3-ASR via vLLM backend (recommended).

    Uses continuous batching and PagedAttention for 2-5x speedup.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None

    def load(self) -> None:
        """Load Qwen3-ASR model via vLLM."""
        from qwen_asr import Qwen3ASRModel  # Lazy import

        asr_config = self.config["asr"]
        vllm_kwargs = {}
        for key in ("gpu_memory_utilization", "max_model_len", "enforce_eager"):
            if key in asr_config:
                vllm_kwargs[key] = asr_config[key]
        vllm_kwargs.setdefault("gpu_memory_utilization", 0.7)
        self.model = Qwen3ASRModel.LLM(
            model=asr_config["model"],
            max_new_tokens=asr_config.get("max_new_tokens", 4096),
            **vllm_kwargs,
        )
        logger.info("Loaded Qwen3-ASR via vLLM backend")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using vLLM backend."""
        language = self.config["asr"].get("language")
        results = self.model.transcribe(audio=audio_path, language=language)
        return results[0].text

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        import torch

        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Unloaded Qwen3-ASR vLLM backend")


class TransformersBackend:
    """Qwen3-ASR via transformers fallback.

    Simpler setup, no vLLM dependency required.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None  # Qwen3ASRModel wrapper

    def load(self) -> None:
        """Load Qwen3-ASR model via transformers."""
        import torch  # Lazy import
        from qwen_asr import Qwen3ASRModel  # Lazy import

        asr_config = self.config["asr"]
        dtype_str = asr_config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_str, torch.bfloat16)

        self.model = Qwen3ASRModel.from_pretrained(
            asr_config["model"],
            torch_dtype=dtype,
            max_new_tokens=asr_config.get("max_new_tokens", 4096),
        )
        # Move inner HF model to GPU without accelerate hooks
        # so .cpu() works cleanly during unload
        if torch.cuda.is_available() and hasattr(self.model, "model"):
            self.model.model = self.model.model.to("cuda:0")
            self.model.device = torch.device("cuda:0")
            self.model.dtype = dtype
        logger.info("Loaded Qwen3-ASR via transformers backend")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using transformers backend."""
        language = self.config["asr"].get("language")
        results = self.model.transcribe(audio=audio_path, language=language)
        return results[0].text

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        import torch

        if self.model is not None:
            # Move inner model to CPU to release VRAM, then delete everything
            inner = getattr(self.model, "model", None)
            if inner is not None:
                inner.cpu()
            del inner
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Unloaded Qwen3-ASR transformers backend")


def _create_backend(config: dict) -> ASRBackend:
    """Create ASR backend based on config.

    Args:
        config: Full pipeline configuration

    Returns:
        ASR backend instance

    Raises:
        ValueError: If unknown backend specified
    """
    backend_name = config["asr"].get("backend", "vllm")
    if backend_name == "vllm":
        return VLLMBackend(config)
    elif backend_name == "transformers":
        return TransformersBackend(config)
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name}")


def run_transcription(
    waveform,  # torch.Tensor - not type-hinted to avoid top-level torch import
    sample_rate: int,
    segments: list[dict],
    config: dict,
    backend: ASRBackend | None = None,
) -> list[dict]:
    """Transcribe each audio segment using Qwen3-ASR.

    Args:
        waveform: Audio waveform tensor (1, N) at target sample rate
        sample_rate: Sample rate in Hz
        segments: List of segment dicts with "start", "end", "speaker" keys
        config: Full pipeline configuration
        backend: Optional pre-configured ASR backend. If None, creates one
            based on config['asr']['backend']

    Returns:
        List of dicts with "start", "end", "speaker", "text" keys
    """
    from src.audio_preprocessing import save_temp_wav, slice_waveform  # Lazy import

    owns_backend = backend is None
    if owns_backend:
        backend = _create_backend(config)

    backend.load()

    results = []
    try:
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            speaker = segment.get("speaker", "UNKNOWN")

            # Slice waveform for this segment
            segment_waveform = slice_waveform(waveform, sample_rate, start, end)

            # Save to temp WAV (Qwen3-ASR accepts file paths)
            temp_path = save_temp_wav(segment_waveform, sample_rate)

            try:
                text = backend.transcribe(str(temp_path))
                logger.debug(
                    f"Segment {i + 1}/{len(segments)} [{start:.1f}s-{end:.1f}s] "
                    f"({speaker}): {text[:80]}..."
                )
            finally:
                # Always clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

            results.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )
    finally:
        backend.unload()

    logger.info(f"Transcribed {len(results)} segments")
    return results
