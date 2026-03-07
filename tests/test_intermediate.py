"""Tests for intermediate result serialization."""

import numpy as np
import torch

from src.intermediate import (
    get_completed_stages,
    load_stage_result,
    save_stage_result,
)


class TestSaveAndLoadStageResult:
    """Test saving and loading stage results."""

    def test_save_and_load_simple_dict(self, tmp_path):
        """Test saving and loading simple dictionary."""
        data = {
            "start": 0.5,
            "end": 3.2,
            "speaker": "Alice",
            "text": "Hello world",
        }

        save_stage_result("stage_0", data, tmp_path)
        loaded = load_stage_result("stage_0", tmp_path)

        assert loaded == data

    def test_save_and_load_with_lists(self, tmp_path):
        """Test saving and loading with nested lists."""
        data = {
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "A"},
                {"start": 1.0, "end": 2.0, "speaker": "B"},
            ]
        }

        save_stage_result("stage_1", data, tmp_path)
        loaded = load_stage_result("stage_1", tmp_path)

        assert loaded == data

    def test_save_and_load_numpy_array(self, tmp_path):
        """Test saving and loading numpy arrays."""
        data = {
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
            "timestamps": np.array([0.5, 1.5, 2.5]),
        }

        save_stage_result("stage_2", data, tmp_path)
        loaded = load_stage_result("stage_2", tmp_path)

        # Arrays converted to lists
        assert loaded["embedding"] == [0.1, 0.2, 0.3, 0.4]
        assert loaded["timestamps"] == [0.5, 1.5, 2.5]

    def test_save_and_load_torch_tensor(self, tmp_path):
        """Test saving and loading torch tensors."""
        data = {
            "tensor": torch.tensor([1.0, 2.0, 3.0]),
            "waveform": torch.randn(1, 100),
        }

        save_stage_result("stage_3", data, tmp_path)
        loaded = load_stage_result("stage_3", tmp_path)

        # Tensors converted to lists
        assert isinstance(loaded["tensor"], list)
        assert len(loaded["waveform"]) == 1
        assert len(loaded["waveform"][0]) == 100

    def test_save_and_load_mixed_types(self, tmp_path):
        """Test saving and loading mixed data types."""
        data = {
            "int": 42,
            "float": 3.14,
            "string": "test",
            "bool": True,
            "none": None,
            "array": np.array([1, 2, 3]),
            "nested": {"inner": torch.tensor([0.5, 0.6])},
        }

        save_stage_result("stage_4", data, tmp_path)
        loaded = load_stage_result("stage_4", tmp_path)

        assert loaded["int"] == 42
        assert loaded["float"] == 3.14
        assert loaded["string"] == "test"
        assert loaded["bool"] is True
        assert loaded["none"] is None
        assert loaded["array"] == [1, 2, 3]
        # Floating point comparison with tolerance
        assert len(loaded["nested"]["inner"]) == 2
        assert abs(loaded["nested"]["inner"][0] - 0.5) < 1e-6
        assert abs(loaded["nested"]["inner"][1] - 0.6) < 1e-6

    def test_load_nonexistent_stage(self, tmp_path):
        """Test loading non-existent stage returns None."""
        result = load_stage_result("nonexistent", tmp_path)
        assert result is None

    def test_save_creates_output_dir(self, tmp_path):
        """Test that save_stage_result creates output directory."""
        output_dir = tmp_path / "subdir" / "nested"
        assert not output_dir.exists()

        data = {"test": "data"}
        save_stage_result("stage_0", data, output_dir)

        assert output_dir.exists()
        assert (output_dir / "stage_0.json").exists()


class TestGetCompletedStages:
    """Test getting completed stages."""

    def test_get_completed_stages_empty_dir(self, tmp_path):
        """Test with empty output directory."""
        completed = get_completed_stages(tmp_path)
        assert completed == []

    def test_get_completed_stages_partial(self, tmp_path):
        """Test with some stages completed."""
        save_stage_result("stage_0", {"test": 0}, tmp_path)
        save_stage_result("stage_1", {"test": 1}, tmp_path)
        # stage_2 and stage_3 not created

        completed = get_completed_stages(tmp_path)
        assert completed == ["stage_0", "stage_1"]

    def test_get_completed_stages_all(self, tmp_path):
        """Test with all stages completed."""
        for i in range(4):
            save_stage_result(f"stage_{i}", {"test": i}, tmp_path)

        completed = get_completed_stages(tmp_path)
        assert completed == ["stage_0", "stage_1", "stage_2", "stage_3"]

    def test_get_completed_stages_nonexistent_dir(self, tmp_path):
        """Test with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        completed = get_completed_stages(nonexistent)
        assert completed == []
