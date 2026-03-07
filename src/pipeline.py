"""Main pipeline orchestrator."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.audio_preprocessing import load_audio
from src.config import load_config
from src.gpu_utils import get_vram_usage
from src.intermediate import get_completed_stages, load_stage_result, save_stage_result


logger = logging.getLogger("asr_pipeline")


def run_pipeline(
    audio_path: str, config: dict | None = None, output_dir: str = "output/"
) -> dict[str, Any]:
    """
    Run the full ASR pipeline serially.

    Flow:
    1. Load and preprocess audio
    2. Stage 0: VAD (silence removal)
    3. Stage 1: Diarization (speaker identification)
    4. Stage 2: ASR (speech-to-text)
    5. Stage 3: LLM post-processing (error correction, summarization)
    6. Format and save output

    Args:
        audio_path: Path to audio file
        config: Configuration dictionary. If None, loads from config.yaml
        output_dir: Output directory for results

    Returns:
        Pipeline result dictionary with all metadata

    Raises:
        FileNotFoundError: If audio file not found
        RuntimeError: If any stage fails
    """
    if config is None:
        config = load_config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    pipeline_result = {
        "audio_file": str(audio_path),
        "output_dir": str(output_dir),
        "stages": {},
    }

    # Check for previous results and resume from next stage
    completed = get_completed_stages(output_dir)
    logger.info(f"Completed stages: {completed}")

    try:
        # Load audio (always done)
        logger.info("Loading audio...")
        vram_before = get_vram_usage()
        t0 = time.perf_counter()
        waveform, sample_rate = load_audio(audio_path)
        pipeline_result["audio_duration_seconds"] = waveform.shape[1] / sample_rate
        timings["audio_load"] = time.perf_counter() - t0
        vram_after = get_vram_usage()
        logger.info(
            f"[audio_load] {timings['audio_load']:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB"
        )

        # Stage 0: VAD
        if "stage_0" not in completed:
            logger.info("Running Stage 0: Voice Activity Detection...")
            vram_before = get_vram_usage()
            t0 = time.perf_counter()
            stage_0_result = _run_stage_0(waveform, sample_rate, config)
            timings["stage_0"] = time.perf_counter() - t0
            vram_after = get_vram_usage()
            logger.info(
                f"[Stage 0: VAD] {timings['stage_0']:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB"
            )
            save_stage_result("stage_0", stage_0_result, output_dir)
        else:
            logger.info("Resuming from Stage 0 (cached)...")
            stage_0_result = load_stage_result("stage_0", output_dir)

        pipeline_result["stages"]["vad"] = stage_0_result

        # Stage 1: Diarization
        if "stage_1" not in completed:
            logger.info("Running Stage 1: Speaker Diarization...")
            vram_before = get_vram_usage()
            t0 = time.perf_counter()
            stage_1_result = _run_stage_1(waveform, sample_rate, config)
            timings["stage_1"] = time.perf_counter() - t0
            vram_after = get_vram_usage()
            logger.info(
                f"[Stage 1: Diarization] {timings['stage_1']:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB"
            )
            save_stage_result("stage_1", stage_1_result, output_dir)
        else:
            logger.info("Resuming from Stage 1 (cached)...")
            stage_1_result = load_stage_result("stage_1", output_dir)

        pipeline_result["stages"]["diarization"] = stage_1_result

        # Stage 2: ASR
        if "stage_2" not in completed:
            logger.info("Running Stage 2: Speech-to-Text...")
            vram_before = get_vram_usage()
            t0 = time.perf_counter()
            stage_2_result = _run_stage_2(waveform, sample_rate, config)
            timings["stage_2"] = time.perf_counter() - t0
            vram_after = get_vram_usage()
            logger.info(
                f"[Stage 2: ASR] {timings['stage_2']:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB"
            )
            save_stage_result("stage_2", stage_2_result, output_dir)
        else:
            logger.info("Resuming from Stage 2 (cached)...")
            stage_2_result = load_stage_result("stage_2", output_dir)

        pipeline_result["stages"]["transcription"] = stage_2_result

        # Stage 3: LLM Post-Processing
        if "stage_3" not in completed:
            logger.info("Running Stage 3: LLM Post-Processing...")
            vram_before = get_vram_usage()
            t0 = time.perf_counter()
            stage_3_result = _run_stage_3(
                stage_2_result, sample_rate, config
            )
            timings["stage_3"] = time.perf_counter() - t0
            vram_after = get_vram_usage()
            logger.info(
                f"[Stage 3: LLM] {timings['stage_3']:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB"
            )
            save_stage_result("stage_3", stage_3_result, output_dir)
        else:
            logger.info("Resuming from Stage 3 (cached)...")
            stage_3_result = load_stage_result("stage_3", output_dir)

        pipeline_result["stages"]["llm_postprocess"] = stage_3_result
        pipeline_result["timings"] = timings
        pipeline_result["total_time_seconds"] = sum(timings.values())

        logger.info(f"Pipeline completed in {pipeline_result['total_time_seconds']:.1f}s")
        return pipeline_result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _run_stage_0(
    waveform, sample_rate: int, config: dict
) -> dict[str, Any]:
    """
    Stage 0: Voice Activity Detection.

    TODO: Implement VAD with Silero VAD v5.
    """
    raise NotImplementedError("Stage 0 (VAD) not yet implemented")


def _run_stage_1(
    waveform, sample_rate: int, config: dict
) -> dict[str, Any]:
    """
    Stage 1: Speaker Diarization.

    TODO: Implement diarization with pyannote.audio and WeSpeaker.
    """
    raise NotImplementedError("Stage 1 (Diarization) not yet implemented")


def _run_stage_2(
    waveform, sample_rate: int, config: dict
) -> dict[str, Any]:
    """
    Stage 2: Speech-to-Text Transcription.

    TODO: Implement ASR with Qwen3-ASR.
    """
    raise NotImplementedError("Stage 2 (ASR) not yet implemented")


def _run_stage_3(
    segments: list[dict], sample_rate: int, config: dict
) -> dict[str, Any]:
    """
    Stage 3: LLM Post-Processing.

    TODO: Implement LLM post-processing with Qwen3.5-9B.
    """
    raise NotImplementedError("Stage 3 (LLM) not yet implemented")
