"""Speaker diarization module with Protocol-based dependency injection."""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger("asr_pipeline")


@runtime_checkable
class DiarizationBackend(Protocol):
    """Protocol for diarization backends."""

    def load(self) -> None: ...
    def run(self, waveform, sample_rate: int) -> list[dict]: ...
    def unload(self) -> None: ...


class PyAnnoteBackend:
    """pyannote speaker-diarization-3.1 backend.

    Loads the pyannote diarization pipeline on demand, runs inference,
    and unloads to free VRAM for subsequent stages.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        diar_cfg = config.get("diarization", {})
        self._model_name = diar_cfg.get("model", "pyannote/speaker-diarization-3.1")
        self._hf_token = diar_cfg.get("hf_token")
        self._num_speakers = diar_cfg.get("num_speakers")
        self._pipeline = None

    def load(self) -> None:
        """Lazy-load pyannote Pipeline and move to CUDA."""
        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline: %s", self._model_name)
        self._pipeline = Pipeline.from_pretrained(
            self._model_name,
            token=self._hf_token,
        )
        if torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))

    def run(self, waveform, sample_rate: int) -> list[dict]:
        """Run diarization and return segments.

        Args:
            waveform: Audio tensor (1, N) on CPU.
            sample_rate: Sample rate in Hz.

        Returns:
            List of {"start": float, "end": float, "speaker": str} dicts.
        """
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        kwargs = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers

        result = self._pipeline(audio_input, **kwargs)

        # pyannote >= 3.3 returns DiarizeOutput dataclass; extract Annotation
        annotation = getattr(result, "speaker_diarization", result)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                }
            )

        logger.info("Diarization produced %d segments", len(segments))
        return segments

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        import torch

        if self._pipeline is not None:
            logger.info("Unloading pyannote diarization pipeline")
            del self._pipeline
            self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def run_diarization(
    waveform,
    sample_rate: int,
    config: dict,
    backend: DiarizationBackend | None = None,
) -> list[dict]:
    """Run speaker diarization on audio.

    If no backend is provided, creates a PyAnnoteBackend from config.
    Calls backend.load(), backend.run(), backend.unload() in sequence.

    Args:
        waveform: Audio tensor (1, N) on CPU.
        sample_rate: Sample rate in Hz.
        config: Pipeline configuration dictionary.
        backend: Optional DiarizationBackend instance for DI/testing.

    Returns:
        List of {"start", "end", "speaker"} dicts sorted by start time.
    """
    if backend is None:
        backend = PyAnnoteBackend(config)

    backend.load()
    try:
        segments = backend.run(waveform, sample_rate)
    finally:
        backend.unload()

    return segments
