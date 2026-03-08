"""Audio loading, resampling, and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _load_with_ffmpeg(path: Path):
    """Load audio via ffmpeg subprocess for formats soundfile can't handle (m4a, mp4, etc.)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(tmp_wav)],
            capture_output=True,
            check=True,
        )
        import soundfile as sf

        data, sample_rate = sf.read(str(tmp_wav), dtype="float32")
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()
    return data, sample_rate


def load_audio(path: str | Path) -> tuple:  # -> tuple[torch.Tensor, int]
    """
    Load audio from file, convert to 16kHz mono, normalize to [-1, 1].

    Supports WAV, FLAC, OGG, and other formats via soundfile.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (waveform, sample_rate) where waveform is (1, N) at 16kHz

    Raises:
        FileNotFoundError: If file not found
        RuntimeError: If audio loading fails
    """
    import soundfile as sf  # Lazy import

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import torch

    # Try soundfile first (fast, no ffmpeg dep), fall back to ffmpeg for m4a/mp4/etc.
    try:
        data, sample_rate = sf.read(str(path), dtype="float32")
    except sf.LibsndfileError:
        data, sample_rate = _load_with_ffmpeg(path)
    # data shape: (N,) for mono, (N, channels) for multi-channel
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    else:
        waveform = torch.from_numpy(data).T  # (channels, N)

    # Convert to mono by averaging channels if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.shape[0] == 0:
        raise ValueError(f"Invalid audio shape: {waveform.shape}")

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import torchaudio  # Lazy import — only needed for resampling

        resample = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resample(waveform)
        sample_rate = 16000

    # Normalize to [-1, 1]
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform, sample_rate


def slice_waveform(
    waveform: torch.Tensor, sample_rate: int, start: float, end: float
) -> torch.Tensor:
    """
    Slice waveform by time in seconds.

    Args:
        waveform: Audio waveform (1, N)
        sample_rate: Sample rate in Hz
        start: Start time in seconds
        end: End time in seconds

    Returns:
        Sliced waveform (1, end_sample - start_sample)
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    return waveform[:, start_sample:end_sample]


def save_temp_wav(waveform: torch.Tensor, sample_rate: int) -> Path:
    """
    Save waveform to temporary WAV file.

    Args:
        waveform: Audio waveform (1, N)
        sample_rate: Sample rate in Hz

    Returns:
        Path to temporary WAV file
    """
    import tempfile

    import soundfile as sf  # Lazy import

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)

    # soundfile expects (N, channels) — transpose from (channels, N)
    sf.write(str(temp_path), waveform.numpy().T, sample_rate)
    return temp_path
