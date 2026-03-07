"""Tests for timestamp utility functions."""

import pytest

from src.timestamp_utils import batch_segments, merge_close_segments, remap_to_original


class TestMergeCloseSegments:
    """Test merge_close_segments function."""

    def test_merge_close_segments_merges_small_gaps(self):
        """Segments with gaps smaller than threshold are merged."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.2, "end": 2.0},
            {"start": 2.1, "end": 3.0},
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 3.0

    def test_merge_close_segments_preserves_large_gaps(self):
        """Segments with gaps larger than threshold remain separate."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 3.0, "end": 4.0},
            {"start": 7.0, "end": 8.0},
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 3
        assert result[0] == {"start": 0.0, "end": 1.0}
        assert result[1] == {"start": 3.0, "end": 4.0}
        assert result[2] == {"start": 7.0, "end": 8.0}

    def test_merge_close_segments_mixed_gaps(self):
        """Some gaps merge, others do not."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.3, "end": 2.0},  # Gap 0.3 -> merge
            {"start": 5.0, "end": 6.0},  # Gap 3.0 -> keep separate
            {"start": 6.1, "end": 7.0},  # Gap 0.1 -> merge with previous
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0
        assert result[1]["start"] == 5.0
        assert result[1]["end"] == 7.0

    def test_merge_close_segments_preserves_other_keys(self):
        """Merged segments preserve non-start/end keys from first segment."""
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "hello"},
            {"start": 1.1, "end": 2.0, "speaker": "Alice", "text": "world"},
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 1
        assert result[0]["speaker"] == "Alice"
        assert result[0]["text"] == "hello"  # First segment's keys preserved
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0

    def test_merge_close_segments_exact_threshold(self):
        """Gap exactly equal to threshold is NOT merged (< not <=)."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.5, "end": 2.5},
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 2

    def test_merge_close_segments_zero_gap(self):
        """Adjacent segments (zero gap) are merged."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0

    def test_merge_close_segments_empty_list(self):
        """Empty input returns empty output."""
        result = merge_close_segments([], gap_threshold=0.5)
        assert result == []

    def test_merge_close_segments_single_segment(self):
        """Single segment returned as-is."""
        segments = [{"start": 1.0, "end": 2.0}]
        result = merge_close_segments(segments, gap_threshold=0.5)
        assert len(result) == 1
        assert result[0] == {"start": 1.0, "end": 2.0}

    def test_merge_close_segments_does_not_mutate_input(self):
        """Input segments are not mutated."""
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.1, "end": 2.0},
        ]
        original = [dict(s) for s in segments]
        merge_close_segments(segments, gap_threshold=0.5)
        assert segments == original


class TestBatchSegments:
    """Test batch_segments function."""

    def test_batch_groups_short_segments(self):
        """Short segments are grouped into a single batch."""
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 3.0, "end": 4.5},
            {"start": 5.0, "end": 7.0},
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_batch_medium_segments_separate(self):
        """Medium segments each get their own batch."""
        segments = [
            {"start": 0.0, "end": 10.0},  # 10s, medium
            {"start": 15.0, "end": 30.0},  # 15s, medium
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_batch_splits_long_segments(self):
        """Long segments are split into sub-segments."""
        segments = [
            {"start": 0.0, "end": 600.0},  # 600s = 10 min, split at 300s
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)
        assert len(batches) == 2
        assert batches[0][0]["start"] == 0.0
        assert batches[0][0]["end"] == 300.0
        assert batches[1][0]["start"] == 300.0
        assert batches[1][0]["end"] == 600.0

    def test_batch_mixed_short_medium_long(self):
        """Mix of short, medium, and long segments batched correctly."""
        segments = [
            {"start": 0.0, "end": 2.0},  # short
            {"start": 3.0, "end": 4.0},  # short
            {"start": 10.0, "end": 25.0},  # medium (15s)
            {"start": 30.0, "end": 31.0},  # short
            {"start": 50.0, "end": 700.0},  # long (650s)
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)

        # Batch 0: two short segments grouped
        assert len(batches[0]) == 2
        # Batch 1: medium segment alone
        assert len(batches[1]) == 1
        assert batches[1][0]["start"] == 10.0
        # Batch 2: short segment alone (flushed before long)
        assert len(batches[2]) == 1
        assert batches[2][0]["start"] == 30.0
        # Remaining: long segment split into 3 chunks (650/300 = ~2.17, so 3 chunks)
        assert len(batches) == 6  # 2 short-batch + 1 medium + 1 short + 3 long-chunks
        assert batches[3][0]["start"] == 50.0
        assert batches[3][0]["end"] == 350.0

    def test_batch_empty_list(self):
        """Empty input returns empty output."""
        assert batch_segments([], max_short=5.0, max_long=300.0) == []

    def test_batch_skips_zero_length_segments(self):
        """Zero-length segments are skipped."""
        segments = [
            {"start": 1.0, "end": 1.0},  # zero length
            {"start": 2.0, "end": 3.0},  # valid short
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0]["start"] == 2.0

    def test_batch_preserves_other_keys(self):
        """Other keys are preserved in batched segments."""
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 3.0, "end": 4.0, "speaker": "B"},
        ]
        batches = batch_segments(segments, max_short=5.0, max_long=300.0)
        assert batches[0][0]["speaker"] == "A"
        assert batches[0][1]["speaker"] == "B"

    def test_batch_does_not_mutate_input(self):
        """Input segments are not mutated."""
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 3.0, "end": 4.0},
        ]
        original = [dict(s) for s in segments]
        batch_segments(segments, max_short=5.0, max_long=300.0)
        assert segments == original


class TestRemapToOriginal:
    """Test remap_to_original function."""

    def test_remap_simple_1_to_1(self):
        """Simple 1:1 mapping where clean and original are the same."""
        timestamp_map = [
            {"original_start": 0.0, "original_end": 5.0, "clean_start": 0.0, "clean_end": 5.0},
        ]
        segments = [
            {"start": 1.0, "end": 3.0},
        ]
        result = remap_to_original(segments, timestamp_map)
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(1.0)
        assert result[0]["end"] == pytest.approx(3.0)

    def test_remap_with_gap_removed(self):
        """Mapping where a silence gap was removed from the middle."""
        # Original: [0-2] speech, [2-4] silence, [4-6] speech
        # Clean:    [0-2] speech, [2-4] speech (no gap)
        timestamp_map = [
            {"original_start": 0.0, "original_end": 2.0, "clean_start": 0.0, "clean_end": 2.0},
            {"original_start": 4.0, "original_end": 6.0, "clean_start": 2.0, "clean_end": 4.0},
        ]
        segments = [
            {"start": 1.0, "end": 1.5},  # In first chunk
            {"start": 2.5, "end": 3.5},  # In second chunk
        ]
        result = remap_to_original(segments, timestamp_map)

        # First segment: clean 1.0 -> original 1.0 (within first chunk, 1:1)
        assert result[0]["start"] == pytest.approx(1.0)
        assert result[0]["end"] == pytest.approx(1.5)

        # Second segment: clean 2.5 -> original 4.5, clean 3.5 -> original 5.5
        assert result[1]["start"] == pytest.approx(4.5)
        assert result[1]["end"] == pytest.approx(5.5)

    def test_remap_preserves_other_keys(self):
        """Other dict keys are preserved after remapping."""
        timestamp_map = [
            {"original_start": 0.0, "original_end": 5.0, "clean_start": 0.0, "clean_end": 5.0},
        ]
        segments = [
            {"start": 1.0, "end": 2.0, "speaker": "Alice", "text": "hello"},
        ]
        result = remap_to_original(segments, timestamp_map)
        assert result[0]["speaker"] == "Alice"
        assert result[0]["text"] == "hello"

    def test_remap_empty_segments(self):
        """Empty segments list returns empty list."""
        timestamp_map = [
            {"original_start": 0.0, "original_end": 5.0, "clean_start": 0.0, "clean_end": 5.0},
        ]
        result = remap_to_original([], timestamp_map)
        assert result == []

    def test_remap_empty_timestamp_map(self):
        """Empty timestamp map returns segments unchanged."""
        segments = [{"start": 1.0, "end": 2.0}]
        result = remap_to_original(segments, [])
        assert result == [{"start": 1.0, "end": 2.0}]

    def test_remap_multiple_gaps(self):
        """Mapping with multiple silence gaps removed."""
        # Original: [0-1] speech, [1-3] silence, [3-5] speech, [5-8] silence, [8-10] speech
        # Clean:    [0-1], [1-3], [3-5]
        timestamp_map = [
            {"original_start": 0.0, "original_end": 1.0, "clean_start": 0.0, "clean_end": 1.0},
            {"original_start": 3.0, "original_end": 5.0, "clean_start": 1.0, "clean_end": 3.0},
            {"original_start": 8.0, "original_end": 10.0, "clean_start": 3.0, "clean_end": 5.0},
        ]
        segments = [
            {"start": 0.5, "end": 0.8},  # First chunk
            {"start": 1.5, "end": 2.5},  # Second chunk
            {"start": 3.5, "end": 4.5},  # Third chunk
        ]
        result = remap_to_original(segments, timestamp_map)

        assert result[0]["start"] == pytest.approx(0.5)
        assert result[0]["end"] == pytest.approx(0.8)

        # clean 1.5 in second chunk: offset = 0.5/2.0 = 0.25 -> orig = 3.0 + 0.25*2.0 = 3.5
        assert result[1]["start"] == pytest.approx(3.5)
        # clean 2.5 in second chunk: offset = 1.5/2.0 = 0.75 -> orig = 3.0 + 0.75*2.0 = 4.5
        assert result[1]["end"] == pytest.approx(4.5)

        # clean 3.5 in third chunk: offset = 0.5/2.0 = 0.25 -> orig = 8.0 + 0.25*2.0 = 8.5
        assert result[2]["start"] == pytest.approx(8.5)
        assert result[2]["end"] == pytest.approx(9.5)

    def test_remap_segment_at_chunk_boundary(self):
        """Segment exactly at chunk start and end.

        When clean_time=2.0 falls on a boundary between two map entries,
        it maps to the start of the second chunk (original_start=5.0).
        This is correct because in the clean audio, time 2.0 is the start
        of the second speech segment.
        """
        timestamp_map = [
            {"original_start": 0.0, "original_end": 2.0, "clean_start": 0.0, "clean_end": 2.0},
            {"original_start": 5.0, "original_end": 7.0, "clean_start": 2.0, "clean_end": 4.0},
        ]
        segments = [
            {"start": 0.0, "end": 1.9},  # Almost entire first chunk (end before boundary)
            {"start": 2.0, "end": 4.0},  # Entire second chunk
        ]
        result = remap_to_original(segments, timestamp_map)

        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(1.9)  # Maps within first chunk
        assert result[1]["start"] == pytest.approx(5.0)
        assert result[1]["end"] == pytest.approx(7.0)

    def test_remap_zero_length_map_entry(self):
        """Zero-length map entry returns original_start."""
        timestamp_map = [
            {"original_start": 3.0, "original_end": 3.0, "clean_start": 0.0, "clean_end": 0.0},
        ]
        segments = [{"start": 0.0, "end": 0.0}]
        result = remap_to_original(segments, timestamp_map)
        assert result[0]["start"] == pytest.approx(3.0)
        assert result[0]["end"] == pytest.approx(3.0)

    def test_remap_does_not_mutate_input(self):
        """Input segments are not mutated."""
        timestamp_map = [
            {"original_start": 0.0, "original_end": 5.0, "clean_start": 0.0, "clean_end": 5.0},
        ]
        segments = [{"start": 1.0, "end": 2.0}]
        original = [dict(s) for s in segments]
        remap_to_original(segments, timestamp_map)
        assert segments == original

    def test_remap_clean_time_past_all_entries(self):
        """Clean time past all map entries clamps to last original_end."""
        timestamp_map = [
            {"original_start": 0.0, "original_end": 2.0, "clean_start": 0.0, "clean_end": 2.0},
        ]
        segments = [{"start": 5.0, "end": 6.0}]
        result = remap_to_original(segments, timestamp_map)
        assert result[0]["start"] == pytest.approx(2.0)
        assert result[0]["end"] == pytest.approx(2.0)
