"""Stage 0: Voice Activity Detection using Silero VAD."""

from src.timestamp_utils import merge_close_segments


def _load_silero():
    """Lazy-load Silero VAD functions. Returns (load_silero_vad, get_speech_timestamps)."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    return load_silero_vad, get_speech_timestamps


def run_vad(waveform, sample_rate: int, config: dict) -> tuple:
    """
    Run Voice Activity Detection to remove silence segments.

    Uses Silero VAD v5 (lazy import). Detects speech segments, merges
    close segments, builds a timestamp map for remapping, and concatenates
    speech into a clean waveform.

    Args:
        waveform: Audio waveform tensor (1, N) at target sample rate
        sample_rate: Sample rate in Hz (typically 16000)
        config: Configuration dictionary with 'vad' section containing:
            - merge_gap_seconds: float, gap threshold for merging segments

    Returns:
        Tuple of:
            - clean_waveform: torch.Tensor (1, M) containing only speech
            - timestamp_map: list of dicts with keys:
                original_start, original_end, clean_start, clean_end
            - speech_segments: list of dicts with 'start' and 'end' keys (in seconds)
    """
    load_silero_vad, get_speech_timestamps = _load_silero()

    # Load Silero VAD model (runs on CPU, ~50MB)
    model = load_silero_vad()

    # Get speech timestamps in seconds
    # Silero expects (N,) or (1, N) tensor; pass the squeezed version
    wav_input = waveform.squeeze(0) if waveform.dim() == 2 else waveform
    speech_timestamps = get_speech_timestamps(wav_input, model, return_seconds=True)

    # Convert to our segment format
    speech_segments = [{"start": ts["start"], "end": ts["end"]} for ts in speech_timestamps]

    # Merge segments where gap is small
    merge_gap = config.get("vad", {}).get("merge_gap_seconds", 0.5)
    speech_segments = merge_close_segments(speech_segments, merge_gap)

    # Build timestamp_map and concatenate speech into clean_waveform
    timestamp_map = []
    clean_parts = []
    clean_offset = 0.0

    for seg in speech_segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        segment_duration = segment_waveform.shape[1] / sample_rate

        timestamp_map.append(
            {
                "original_start": seg["start"],
                "original_end": seg["end"],
                "clean_start": clean_offset,
                "clean_end": clean_offset + segment_duration,
            }
        )

        clean_parts.append(segment_waveform)
        clean_offset += segment_duration

    # Concatenate all speech segments
    import torch

    if clean_parts:
        clean_waveform = torch.cat(clean_parts, dim=1)
    else:
        clean_waveform = torch.zeros(1, 0, dtype=waveform.dtype)

    # Cleanup VAD model (CPU-only, minimal memory, but be consistent)
    del model

    return clean_waveform, timestamp_map, speech_segments
