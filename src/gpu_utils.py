"""GPU memory management utilities."""

import gc
from collections.abc import Generator
from contextlib import contextmanager


def force_gpu_cleanup() -> None:
    """
    Force garbage collection and GPU cache clearing.

    Callers must ``del`` their own model references before calling this.
    Sequence: caller ``del model`` → ``force_gpu_cleanup()``.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_usage() -> float:
    """Return current VRAM usage in MB."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def check_vram_available(required_mb: float) -> None:
    """
    Check if enough VRAM is available, raise if not.

    Args:
        required_mb: Required VRAM in MB

    Raises:
        RuntimeError: If insufficient VRAM
    """
    import torch

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        free = total_mem - allocated
        if free < required_mb:
            raise RuntimeError(
                f"Insufficient VRAM: {free:.0f}MB available, {required_mb:.0f}MB required"
            )


@contextmanager
def gpu_stage(stage_name: str, required_mb: float) -> Generator:
    """
    Context manager for GPU stage execution.

    Checks VRAM on entry, ensures cleanup on exit.

    Args:
        stage_name: Name of the stage for logging
        required_mb: Required VRAM in MB

    Yields:
        None

    Raises:
        RuntimeError: If insufficient VRAM
    """
    vram_before = get_vram_usage()
    check_vram_available(required_mb)
    try:
        yield
    finally:
        force_gpu_cleanup()
        vram_after = get_vram_usage()
        print(f"[{stage_name}] VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB")


def log_vram(stage_name: str, logger=None) -> None:
    """
    Log current VRAM usage.

    Args:
        stage_name: Name of the stage
        logger: Optional logger instance. If None, uses print()
    """
    usage = get_vram_usage()
    msg = f"[{stage_name}] VRAM: {usage:.0f}MB"
    if logger:
        logger.info(msg)
    else:
        print(msg)
