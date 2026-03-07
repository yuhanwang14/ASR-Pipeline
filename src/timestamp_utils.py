"""Pure utility functions for timestamp operations."""


def remap_to_original(segments: list[dict], timestamp_map: list[dict]) -> list[dict]:
    """
    Map clean audio positions back to original positions using timestamp_map.

    Each entry in timestamp_map has: original_start, original_end, clean_start, clean_end.
    For each segment, find which map entries overlap with its clean-audio position
    and compute the corresponding original time.

    Args:
        segments: List of dicts with 'start' and 'end' keys (in clean audio time)
        timestamp_map: List of dicts mapping clean audio ranges to original ranges

    Returns:
        New list of segments with start/end remapped to original audio time.
        All other dict keys are preserved.
    """
    if not segments or not timestamp_map:
        return [dict(seg) for seg in segments]

    result = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["start"] = _clean_to_original(seg["start"], timestamp_map)
        new_seg["end"] = _clean_to_original(seg["end"], timestamp_map)
        result.append(new_seg)
    return result


def _clean_to_original(clean_time: float, timestamp_map: list[dict]) -> float:
    """Convert a single clean-audio timestamp to original-audio timestamp.

    When clean_time falls exactly on a boundary between two map entries,
    it is assigned to the later entry (i.e. the start of the next chunk).
    """
    last_idx = len(timestamp_map) - 1
    for i, entry in enumerate(timestamp_map):
        # Use exclusive upper bound except for the very last entry
        if i < last_idx:
            in_range = entry["clean_start"] <= clean_time < entry["clean_end"]
        else:
            in_range = entry["clean_start"] <= clean_time <= entry["clean_end"]

        if in_range:
            clean_duration = entry["clean_end"] - entry["clean_start"]
            if clean_duration == 0:
                return entry["original_start"]
            offset_ratio = (clean_time - entry["clean_start"]) / clean_duration
            original_duration = entry["original_end"] - entry["original_start"]
            return entry["original_start"] + offset_ratio * original_duration

    # If clean_time is past all map entries, clamp to the end of the last entry
    if timestamp_map and clean_time > timestamp_map[-1]["clean_end"]:
        return timestamp_map[-1]["original_end"]

    # If clean_time is before all map entries, clamp to the start of the first entry
    if timestamp_map and clean_time < timestamp_map[0]["clean_start"]:
        return timestamp_map[0]["original_start"]

    return clean_time


def merge_close_segments(segments: list[dict], gap_threshold: float) -> list[dict]:
    """
    Merge segments where the gap between consecutive segments is less than gap_threshold seconds.

    Each segment has 'start' and 'end' keys (floats in seconds).
    When merging, the resulting segment spans from the earliest start to the latest end.
    Other dict keys from the first segment in each merged group are preserved.

    Args:
        segments: List of dicts with at least 'start' and 'end' keys
        gap_threshold: Maximum gap in seconds to merge across

    Returns:
        New list of merged segments
    """
    if not segments:
        return []

    result = [dict(segments[0])]

    for seg in segments[1:]:
        gap = seg["start"] - result[-1]["end"]
        if gap < gap_threshold:
            # Extend the current segment's end
            result[-1]["end"] = max(result[-1]["end"], seg["end"])
        else:
            result.append(dict(seg))

    return result


def batch_segments(
    segments: list[dict], max_short: float = 5.0, max_long: float = 300.0
) -> list[list[dict]]:
    """
    Group short segments into batches and split long segments at natural boundaries.

    Short segments (duration < max_short) are grouped together into batches.
    Long segments (duration > max_long) are split into sub-segments of max_long duration.
    Medium segments (max_short <= duration <= max_long) each get their own batch.

    Args:
        segments: List of dicts with 'start' and 'end' keys
        max_short: Maximum duration in seconds to be considered "short"
        max_long: Maximum duration in seconds before splitting

    Returns:
        List of batches, each batch is a list of segments
    """
    if not segments:
        return []

    batches = []
    current_short_batch = []

    for seg in segments:
        duration = seg["end"] - seg["start"]

        if duration <= 0:
            # Skip zero-length or negative segments
            continue

        if duration < max_short:
            # Accumulate short segments into a batch
            current_short_batch.append(dict(seg))
        else:
            # Flush any accumulated short segments first
            if current_short_batch:
                batches.append(current_short_batch)
                current_short_batch = []

            if duration > max_long:
                # Split long segment into chunks of max_long
                batches.extend(_split_long_segment(seg, max_long))
            else:
                # Medium segment gets its own batch
                batches.append([dict(seg)])

    # Flush remaining short segments
    if current_short_batch:
        batches.append(current_short_batch)

    return batches


def _split_long_segment(seg: dict, max_long: float) -> list[list[dict]]:
    """Split a long segment into sub-segments of at most max_long duration."""
    result = []
    current_start = seg["start"]
    end = seg["end"]

    while current_start < end:
        chunk_end = min(current_start + max_long, end)
        chunk = dict(seg)
        chunk["start"] = current_start
        chunk["end"] = chunk_end
        result.append([chunk])
        current_start = chunk_end

    return result
