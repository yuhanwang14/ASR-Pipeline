"""Tests for GPU utilities."""

from unittest.mock import MagicMock, patch

import pytest

from src.gpu_utils import (
    check_vram_available,
    force_gpu_cleanup,
    get_vram_usage,
    gpu_stage,
    log_vram,
)


class TestForceGpuCleanup:
    """Test force_gpu_cleanup function."""

    def test_calls_gc_collect(self):
        """Test that force_gpu_cleanup calls gc.collect."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("gc.collect") as mock_collect:
                force_gpu_cleanup()
                mock_collect.assert_called_once()

    def test_with_cuda(self):
        """Test force_gpu_cleanup with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("torch.cuda.synchronize") as mock_sync:
                    with patch("gc.collect"):
                        force_gpu_cleanup()
                        mock_empty.assert_called_once()
                        mock_sync.assert_called_once()


class TestGetVramUsage:
    """Test get_vram_usage function."""

    def test_get_vram_usage_no_cuda(self):
        """Test VRAM usage when CUDA not available."""
        with patch("torch.cuda.is_available", return_value=False):
            usage = get_vram_usage()
            assert usage == 0.0

    def test_get_vram_usage_with_cuda(self):
        """Test VRAM usage when CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 512):
                usage = get_vram_usage()
                assert usage == 512.0


class TestCheckVramAvailable:
    """Test check_vram_available function."""

    def test_check_vram_sufficient(self):
        """Test when VRAM is sufficient."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_properties") as mock_props:
                mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024
                with patch("torch.cuda.memory_allocated", return_value=0):
                    # Should not raise
                    check_vram_available(1000)

    def test_check_vram_insufficient(self):
        """Test when VRAM is insufficient."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_properties") as mock_props:
                mock_props.return_value.total_memory = 1 * 1024 * 1024 * 1024
                with patch("torch.cuda.memory_allocated", return_value=0):
                    with pytest.raises(RuntimeError, match="Insufficient VRAM"):
                        check_vram_available(2000)


class TestGpuStageContext:
    """Test gpu_stage context manager."""

    def test_gpu_stage_checks_vram(self):
        """Test that gpu_stage checks VRAM on entry."""
        with patch("src.gpu_utils.check_vram_available") as mock_check:
            with patch("src.gpu_utils.get_vram_usage", return_value=100.0):
                with patch("src.gpu_utils.force_gpu_cleanup"):
                    with gpu_stage("test", 500):
                        pass
                    mock_check.assert_called_once_with(500)

    def test_gpu_stage_raises_on_insufficient_vram(self):
        """Test that gpu_stage raises on insufficient VRAM."""
        with patch(
            "src.gpu_utils.check_vram_available",
            side_effect=RuntimeError("Insufficient"),
        ):
            with pytest.raises(RuntimeError):
                with gpu_stage("test", 500):
                    pass

    def test_gpu_stage_calls_cleanup_on_exit(self):
        """Test that gpu_stage calls force_gpu_cleanup in finally block."""
        with patch("src.gpu_utils.check_vram_available"):
            with patch("src.gpu_utils.get_vram_usage", return_value=100.0):
                with patch("src.gpu_utils.force_gpu_cleanup") as mock_cleanup:
                    with gpu_stage("test", 500):
                        pass
                    mock_cleanup.assert_called_once()

    def test_gpu_stage_logs_vram_after_exit(self, capsys):
        """Test that gpu_stage logs VRAM after exiting."""
        with patch("src.gpu_utils.check_vram_available"):
            with patch("src.gpu_utils.get_vram_usage", side_effect=[100.0, 150.0]):
                with patch("src.gpu_utils.force_gpu_cleanup"):
                    with gpu_stage("test", 500):
                        pass
                    captured = capsys.readouterr()
                    assert "test" in captured.out
                    assert "100" in captured.out
                    assert "150" in captured.out


class TestLogVram:
    """Test log_vram function."""

    def test_log_vram_prints_default(self, capsys):
        """Test log_vram prints to stdout."""
        with patch("src.gpu_utils.get_vram_usage", return_value=256.0):
            log_vram("test_stage")
            captured = capsys.readouterr()
            assert "test_stage" in captured.out
            assert "256" in captured.out

    def test_log_vram_with_logger(self):
        """Test log_vram with logger instance."""
        logger = MagicMock()
        with patch("src.gpu_utils.get_vram_usage", return_value=256.0):
            log_vram("test_stage", logger)
            logger.info.assert_called_once()
            assert "test_stage" in logger.info.call_args[0][0]
