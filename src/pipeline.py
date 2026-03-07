"""Main pipeline orchestrator."""

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
            stage_3_result = _run_stage_3(stage_2_result, sample_rate, config)
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

        # Expose final segments and summary at top level for output formatters
        pipeline_result["segments"] = stage_3_result.get("segments", [])
        pipeline_result["summary"] = stage_3_result.get("summary")
        pipeline_result["warnings"] = stage_3_result.get("warnings", [])
        pipeline_result["timings"] = timings
        pipeline_result["total_time_seconds"] = sum(timings.values())

        logger.info(f"Pipeline completed in {pipeline_result['total_time_seconds']:.1f}s")
        return pipeline_result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _run_stage_0(waveform, sample_rate: int, config: dict) -> dict[str, Any]:
    """Stage 0: Voice Activity Detection."""
    from src.vad import run_vad

    clean_waveform, timestamp_map, speech_segments = run_vad(waveform, sample_rate, config)
    return {
        "speech_segments": speech_segments,
        "timestamp_map": timestamp_map,
        "clean_waveform_samples": clean_waveform.shape[1],
    }


def _run_stage_1(waveform, sample_rate: int, config: dict) -> dict[str, Any]:
    """Stage 1: Speaker Diarization."""
    from src.diarization import run_diarization
    from src.gpu_utils import gpu_stage
    from src.speaker_registry import match_speakers

    with gpu_stage("Diarization", required_mb=1000):
        segments = run_diarization(waveform, sample_rate, config)

    # Match speakers against enrolled profiles
    segments = match_speakers(segments, config)

    return {"segments": segments}


def _run_stage_2(waveform, sample_rate: int, config: dict) -> dict[str, Any]:
    """Stage 2: Speech-to-Text Transcription."""
    from src.gpu_utils import gpu_stage
    from src.transcription import run_transcription

    # Use diarization segments from stage 1
    stage_1 = load_stage_result("stage_1", config.get("output", {}).get("output_dir", "output/"))
    segments = stage_1["segments"] if stage_1 else []

    with gpu_stage("ASR", required_mb=1500):
        transcript_segments = run_transcription(waveform, sample_rate, segments, config)

    return {"segments": transcript_segments}


def _run_stage_3(stage_2_result, sample_rate: int, config: dict) -> dict[str, Any]:
    """Stage 3: LLM Post-Processing."""
    from src.gpu_utils import gpu_stage
    from src.llm_postprocess import run_llm_postprocess

    segments = stage_2_result.get("segments", [])

    with gpu_stage("LLM", required_mb=5000):
        result = run_llm_postprocess(segments, config)

    return result
