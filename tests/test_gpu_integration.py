"""GPU integration tests — verify VRAM stays under 8GB per stage.

Run: uv run pytest -m gpu -v --tb=long
Requires: CUDA GPU, HF_TOKEN env var, models downloaded locally.
"""

import gc
import os
import threading
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
HF_TOKEN = os.environ.get("HF_TOKEN")
GGUF_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-9B-Q4_K_M.gguf"
TEST_AUDIO = Path(__file__).resolve().parent / "fixtures" / "Zoom - Feb 27.wav"

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
requires_hf_token = pytest.mark.skipif(not HF_TOKEN, reason="HF_TOKEN env var not set")
requires_gguf = pytest.mark.skipif(not GGUF_PATH.exists(), reason=f"GGUF not found: {GGUF_PATH}")
requires_test_audio = pytest.mark.skipif(
    not TEST_AUDIO.exists(), reason=f"Test audio not found: {TEST_AUDIO}"
)

# ---------------------------------------------------------------------------
# VRAM measurement helpers
# ---------------------------------------------------------------------------

# VRAM thresholds in MB
VRAM_CEILING_VAD = 200
VRAM_CEILING_DIARIZATION = 3 * 1024  # 3 GB
VRAM_CEILING_ASR = 4 * 1024  # 4 GB
VRAM_CEILING_LLM = 7 * 1024  # 7 GB
VRAM_CEILING_AFTER_UNLOAD = 512
VRAM_CEILING_PIPELINE = 8 * 1024  # 8 GB


def get_gpu_memory_used_mb() -> float:
    """Get total GPU memory used in MB (driver-level, all allocators)."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


class VRAMMonitor:
    """Context manager that polls GPU memory usage and records peak.

    Uses torch.cuda.mem_get_info() which captures ALL GPU memory usage
    (PyTorch, vLLM, llama-cpp, etc.), not just PyTorch allocations.

    Usage:
        with VRAMMonitor() as monitor:
            # ... load and run model ...
        print(f"Peak VRAM delta: {monitor.peak_delta_mb:.0f} MB")
    """

    def __init__(self, poll_interval: float = 0.05):
        self.poll_interval = poll_interval
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self.peak_delta_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self):
        while not self._stop.is_set():
            used = get_gpu_memory_used_mb()
            if used > self.peak_mb:
                self.peak_mb = used
                self.peak_delta_mb = self.peak_mb - self.baseline_mb
            self._stop.wait(self.poll_interval)

    def __enter__(self):
        self.baseline_mb = get_gpu_memory_used_mb()
        self.peak_mb = self.baseline_mb
        self.peak_delta_mb = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # One final measurement
        used = get_gpu_memory_used_mb()
        if used > self.peak_mb:
            self.peak_mb = used
            self.peak_delta_mb = self.peak_mb - self.baseline_mb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def vram_cleanup():
    """Ensure clean GPU state before and after each test."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def test_audio_path() -> Path:
    """Path to the Zoom test recording."""
    if not TEST_AUDIO.exists():
        pytest.skip(f"Test audio not found: {TEST_AUDIO}")
    return TEST_AUDIO


@pytest.fixture
def short_waveform(test_audio_path):
    """Load first 30 seconds of test audio for per-stage tests."""
    from src.audio_preprocessing import load_audio

    waveform, sample_rate = load_audio(str(test_audio_path))
    # Take first 30 seconds
    max_samples = 30 * sample_rate
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform, sample_rate


@pytest.fixture
def gpu_config(tmp_path) -> dict:
    """Pipeline config for GPU integration tests."""
    return {
        "audio": {
            "target_sample_rate": 16000,
            "target_channels": 1,
        },
        "vad": {
            "model": "silero_vad",
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "merge_gap_seconds": 0.5,
        },
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "hf_token": HF_TOKEN,
            "num_speakers": None,
            "speaker_profiles_dir": str(tmp_path / "speaker_profiles"),
            "match_threshold": 0.75,
        },
        "asr": {
            "model": "Qwen/Qwen3-ASR-1.7B",
            "backend": "vllm",
            "dtype": "bfloat16",
            "max_new_tokens": 4096,
            "language": None,
            "max_segment_duration": 300,
            "gpu_memory_utilization": 0.35,  # Low for testing — measures actual model footprint
            "flash_attention": True,
        },
        "llm": {
            "model_path": str(GGUF_PATH),
            "model_id": "Qwen/Qwen3.5-9B",
            "gguf_url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
            "backend": "llama-cpp",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "tasks": {
                "speaker_correction": True,
                "text_correction": True,
                "summarization": False,  # Skip summarization to save time
            },
        },
        "output": {
            "formats": ["json"],
            "output_dir": str(tmp_path / "output"),
            "save_intermediate": True,
        },
    }


# ---------------------------------------------------------------------------
# Per-stage VRAM tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_cuda
@requires_test_audio
class TestStage0VAD:
    """Stage 0: Silero VAD runs on CPU, should use minimal GPU memory."""

    def test_vad_vram(self, short_waveform, gpu_config):
        from src.vad import run_vad

        waveform, sample_rate = short_waveform

        with VRAMMonitor() as monitor:
            clean_waveform, timestamp_map, speech_segments = run_vad(
                waveform, sample_rate, gpu_config
            )

        print(f"\n[Stage 0: VAD] Peak VRAM delta: {monitor.peak_delta_mb:.0f} MB")
        assert monitor.peak_delta_mb < VRAM_CEILING_VAD, (
            f"VAD VRAM {monitor.peak_delta_mb:.0f}MB exceeded {VRAM_CEILING_VAD}MB ceiling"
        )

        # Sanity: VAD should find some speech
        assert len(speech_segments) > 0, "VAD found no speech in test audio"
        assert clean_waveform.shape[1] > 0, "Clean waveform is empty"


@pytest.mark.gpu
@requires_cuda
@requires_hf_token
@requires_test_audio
class TestStage1Diarization:
    """Stage 1: pyannote diarization — expected ~1-2GB VRAM."""

    def test_diarization_vram(self, short_waveform, gpu_config):
        from src.diarization import run_diarization

        waveform, sample_rate = short_waveform

        with VRAMMonitor() as monitor:
            segments = run_diarization(waveform, sample_rate, gpu_config)

        peak = monitor.peak_delta_mb
        print(f"\n[Stage 1: Diarization] Peak VRAM delta: {peak:.0f} MB")
        assert peak < VRAM_CEILING_DIARIZATION, (
            f"Diarization VRAM {peak:.0f}MB exceeded {VRAM_CEILING_DIARIZATION}MB ceiling"
        )

        # Verify cleanup
        after_unload = get_gpu_memory_used_mb() - monitor.baseline_mb
        print(f"[Stage 1: Diarization] VRAM after unload: {after_unload:.0f} MB above baseline")
        assert after_unload < VRAM_CEILING_AFTER_UNLOAD, (
            f"VRAM after unload {after_unload:.0f}MB exceeded {VRAM_CEILING_AFTER_UNLOAD}MB"
        )

        # Sanity: should find at least 1 speaker segment
        assert len(segments) > 0, "Diarization found no speaker segments"
        assert all("speaker" in s for s in segments), "Segments missing 'speaker' key"


@pytest.mark.gpu
@requires_cuda
@requires_test_audio
class TestStage2ASR:
    """Stage 2: Qwen3-ASR-1.7B — expected ~2GB VRAM.

    Uses gpu_memory_utilization=0.35 in test config so vLLM doesn't
    pre-allocate more than needed. This measures actual model footprint.
    """

    def test_asr_vram(self, short_waveform, gpu_config):
        from src.transcription import run_transcription

        waveform, sample_rate = short_waveform

        # Create minimal diarization segments covering the 30s clip
        segments = [{"start": 0.0, "end": 15.0, "speaker": "SPEAKER_00"}]

        with VRAMMonitor() as monitor:
            transcript_segments = run_transcription(waveform, sample_rate, segments, gpu_config)

        peak = monitor.peak_delta_mb
        print(f"\n[Stage 2: ASR] Peak VRAM delta: {peak:.0f} MB")
        assert peak < VRAM_CEILING_ASR, (
            f"ASR VRAM {peak:.0f}MB exceeded {VRAM_CEILING_ASR}MB ceiling"
        )

        # Verify cleanup
        after_unload = get_gpu_memory_used_mb() - monitor.baseline_mb
        print(f"[Stage 2: ASR] VRAM after unload: {after_unload:.0f} MB above baseline")
        assert after_unload < VRAM_CEILING_AFTER_UNLOAD, (
            f"VRAM after unload {after_unload:.0f}MB exceeded {VRAM_CEILING_AFTER_UNLOAD}MB"
        )

        # Sanity: transcription should produce text
        assert len(transcript_segments) > 0, "ASR produced no transcript segments"
        assert all("text" in s and len(s["text"]) > 0 for s in transcript_segments), (
            "Transcript segments missing 'text'"
        )


@pytest.mark.gpu
@requires_cuda
@requires_gguf
class TestStage3LLM:
    """Stage 3: Qwen3.5-9B Q4_K_M via llama-cpp — expected ~5.5GB VRAM."""

    def test_llm_vram(self, gpu_config):
        from src.llm_postprocess import run_llm_postprocess

        # Sample transcript segments to feed the LLM
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker": "SPEAKER_00",
                "text": "今天我们来讨论一下 project timeline",
            },
            {
                "start": 3.0,
                "end": 6.0,
                "speaker": "SPEAKER_01",
                "text": "OK let me pull up the schedule",
            },
            {
                "start": 6.0,
                "end": 9.0,
                "speaker": "SPEAKER_00",
                "text": "我觉得 deadline 可以往后推一周",
            },
        ]

        with VRAMMonitor() as monitor:
            result = run_llm_postprocess(segments, gpu_config)

        peak = monitor.peak_delta_mb
        print(f"\n[Stage 3: LLM] Peak VRAM delta: {peak:.0f} MB")
        assert peak < VRAM_CEILING_LLM, (
            f"LLM VRAM {peak:.0f}MB exceeded {VRAM_CEILING_LLM}MB ceiling"
        )

        # Verify cleanup
        after_unload = get_gpu_memory_used_mb() - monitor.baseline_mb
        print(f"[Stage 3: LLM] VRAM after unload: {after_unload:.0f} MB above baseline")
        assert after_unload < VRAM_CEILING_AFTER_UNLOAD, (
            f"VRAM after unload {after_unload:.0f}MB exceeded {VRAM_CEILING_AFTER_UNLOAD}MB"
        )

        # Sanity: result should have segments
        assert "segments" in result, "LLM result missing 'segments'"
        assert len(result["segments"]) > 0, "LLM produced no output segments"
