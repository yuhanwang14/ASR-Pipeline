"""Output formatting for pipeline results (JSON, SRT, RTTM, TXT)."""

import json
import logging
from pathlib import Path

logger = logging.getLogger("asr_pipeline.output")


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds (non-negative).

    Returns:
        Formatted SRT timestamp string.
    """
    seconds = max(0.0, seconds)
    total_millis = round(seconds * 1000)
    millis = total_millis % 1000
    total_secs = total_millis // 1000
    secs = total_secs % 60
    minutes = (total_secs // 60) % 60
    hours = total_secs // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_json(result: dict, metadata: dict) -> str:
    """Format pipeline result as JSON string.

    Schema:
    {
        "metadata": {
            "audio_file": str,
            "duration_seconds": float,
            "num_speakers": int,
            "processing_time_seconds": float
        },
        "speakers": [{"id": str, "name": str|null, "confidence": float|null}],
        "segments": [{"start": float, "end": float, "speaker": str, "text": str}],
    }

    Args:
        result: Pipeline result dictionary from run_pipeline().
        metadata: Additional metadata to include (audio_file, duration, etc.).

    Returns:
        JSON string with ensure_ascii=False for Chinese text support.
    """
    # Extract unique speakers from segments
    segments = result.get("segments", [])
    speaker_ids = []
    seen = set()
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk not in seen:
            seen.add(spk)
            speaker_ids.append(spk)

    speakers = []
    for spk_id in speaker_ids:
        speakers.append(
            {
                "id": spk_id,
                "name": spk_id if not spk_id.startswith("SPEAKER_") else None,
                "confidence": None,
            }
        )

    output = {
        "metadata": {
            "audio_file": metadata.get("audio_file", ""),
            "duration_seconds": metadata.get("duration_seconds", 0.0),
            "num_speakers": len(speakers),
            "processing_time_seconds": metadata.get("processing_time_seconds", 0.0),
        },
        "speakers": speakers,
        "segments": [
            {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg.get("text", ""),
            }
            for seg in segments
        ],
    }

    return json.dumps(output, indent=2, ensure_ascii=False)


def to_srt(segments: list[dict]) -> str:
    """Format segments as SRT subtitle format.

    Example output:
        1
        00:00:00,500 --> 00:00:03,200
        [Alice] Hello world

        2
        00:00:03,200 --> 00:00:05,000
        [Bob] Hi there

    Args:
        segments: List of segment dicts with start, end, speaker, text.

    Returns:
        SRT-formatted string.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = _format_srt_time(seg.get("start", 0.0))
        end_ts = _format_srt_time(seg.get("end", 0.0))
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"[{speaker}] {text}")
        lines.append("")  # blank line separator

    return "\n".join(lines)


def to_rttm(segments: list[dict], audio_filename: str) -> str:
    """Format segments as RTTM (Rich Transcription Time Marked).

    Format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Args:
        segments: List of segment dicts with start, end, speaker.
        audio_filename: Audio filename (without path) for the RTTM file field.

    Returns:
        RTTM-formatted string.
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = end - start
        speaker = seg.get("speaker", "UNKNOWN")
        lines.append(
            f"SPEAKER {audio_filename} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines)


def to_txt(segments: list[dict]) -> str:
    """Format segments as plain text with timestamps.

    Format: [HH:MM:SS] Speaker: text

    Args:
        segments: List of segment dicts with start, speaker, text.

    Returns:
        Plain text formatted string.
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0.0)
        hours = int(start // 3600)
        minutes = int((start % 3600) // 60)
        secs = int(start % 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"[{timestamp}] {speaker}: {text}")
    return "\n".join(lines)


def save_outputs(result: dict, config: dict, audio_path: str) -> list[str]:
    """Save all configured output formats to output_dir.

    Uses config['output']['formats'] to determine which formats to save.
    Uses config['output']['output_dir'] for the output directory.

    Args:
        result: Pipeline result dictionary from run_pipeline().
        config: Pipeline configuration dictionary.
        audio_path: Path to the original audio file.

    Returns:
        List of saved file paths (absolute).
    """
    output_cfg = config.get("output", {})
    formats = output_cfg.get("formats", ["json", "srt", "txt"])
    output_dir = Path(output_cfg.get("output_dir", "output/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_name = Path(audio_path).stem
    audio_filename = Path(audio_path).name

    # Build segments list from result
    segments = result.get("segments", [])

    # Build metadata for JSON output
    metadata = {
        "audio_file": str(audio_path),
        "duration_seconds": result.get("audio_duration_seconds", 0.0),
        "processing_time_seconds": result.get("total_time_seconds", 0.0),
    }

    saved_paths = []

    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "json":
            content = to_json(result, metadata)
            out_path = output_dir / f"{audio_name}.json"
        elif fmt == "srt":
            content = to_srt(segments)
            out_path = output_dir / f"{audio_name}.srt"
        elif fmt == "rttm":
            content = to_rttm(segments, audio_filename)
            out_path = output_dir / f"{audio_name}.rttm"
        elif fmt == "txt":
            content = to_txt(segments)
            out_path = output_dir / f"{audio_name}.txt"
        else:
            logger.warning("Unknown output format '%s', skipping.", fmt)
            continue

        out_path.write_text(content, encoding="utf-8")
        saved_paths.append(str(out_path.resolve()))
        logger.info("Saved %s output: %s", fmt.upper(), out_path)

    return saved_paths
