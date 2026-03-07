"""Tests for audio preprocessing."""

import pytest
import torch

from src.audio_preprocessing import load_audio, save_temp_wav, slice_waveform

# Mark audio loading tests as integration since they require ffmpeg
pytestmark = pytest.mark.integration


class TestLoadAudio:
    """Test audio loading."""

    def test_load_audio_file_not_found(self):
        """Test error when audio file not found."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent.wav")

    def test_load_audio_mono(self, tmp_path):
        """Test loading mono audio."""
        audio_file = tmp_path / "test.wav"

        # Create a simple mono audio file
        sample_rate = 16000
        waveform = torch.sin(2 * 3.14159 * 440 * torch.arange(sample_rate) / sample_rate)
        waveform = waveform.unsqueeze(0)  # (1, N)

        import torchaudio

        torchaudio.save(str(audio_file), waveform, sample_rate)

        # Load and verify
        loaded, sr = load_audio(audio_file)
        assert loaded.shape[0] == 1  # Mono
        assert sr == 16000
        assert loaded.dtype == torch.float32
        assert loaded.abs().max() <= 1.0  # Normalized

    def test_load_audio_stereo_to_mono(self, tmp_path):
        """Test converting stereo to mono."""
        audio_file = tmp_path / "test_stereo.wav"

        # Create stereo audio
        sample_rate = 16000
        waveform = torch.randn(2, sample_rate)  # 2 channels, 1 second

        import torchaudio

        torchaudio.save(str(audio_file), waveform, sample_rate)

        # Load and verify mono conversion
        loaded, sr = load_audio(audio_file)
        assert loaded.shape[0] == 1  # Converted to mono
        assert sr == 16000

    def test_load_audio_resample(self, tmp_path):
        """Test resampling to 16kHz."""
        audio_file = tmp_path / "test_resample.wav"

        # Create 44.1kHz audio
        sample_rate = 44100
        waveform = torch.randn(1, sample_rate)

        import torchaudio

        torchaudio.save(str(audio_file), waveform, sample_rate)

        # Load and verify resampling
        loaded, sr = load_audio(audio_file)
        assert sr == 16000
        # Rough check: 44.1kHz -> 16kHz should reduce samples by ~2.75x
        assert loaded.shape[1] < waveform.shape[1]

    def test_load_audio_normalization(self, tmp_path):
        """Test audio normalization to [-1, 1]."""
        audio_file = tmp_path / "test_norm.wav"

        # Create audio with large values
        sample_rate = 16000
        waveform = torch.randn(1, sample_rate) * 10.0  # Large amplitude

        import torchaudio

        torchaudio.save(str(audio_file), waveform, sample_rate)

        # Load and verify normalization
        loaded, sr = load_audio(audio_file)
        assert loaded.abs().max() <= 1.0
        assert loaded.abs().min() >= 0.0


class TestSliceWaveform:
    """Test waveform slicing."""

    def test_slice_waveform_middle(self):
        """Test slicing middle of waveform."""
        waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        sample_rate = 16000

        # Slice 0.25s to 0.75s
        sliced = slice_waveform(waveform, sample_rate, 0.25, 0.75)
        assert sliced.shape[0] == 1
        # Should be ~0.5 seconds = ~8000 samples
        assert 7500 < sliced.shape[1] < 8500

    def test_slice_waveform_start(self):
        """Test slicing from start."""
        waveform = torch.randn(1, 16000)
        sample_rate = 16000

        sliced = slice_waveform(waveform, sample_rate, 0.0, 0.5)
        assert sliced.shape[0] == 1
        assert 7500 < sliced.shape[1] < 8500

    def test_slice_waveform_end(self):
        """Test slicing to end."""
        waveform = torch.randn(1, 16000)
        sample_rate = 16000

        sliced = slice_waveform(waveform, sample_rate, 0.5, 1.0)
        assert sliced.shape[0] == 1
        assert 7500 < sliced.shape[1] < 8500


class TestSaveTempWav:
    """Test temporary WAV saving."""

    def test_save_temp_wav_creates_file(self):
        """Test that save_temp_wav creates a valid temp file."""
        waveform = torch.sin(2 * 3.14159 * 440 * torch.arange(16000) / 16000).unsqueeze(0)
        sample_rate = 16000

        temp_path = save_temp_wav(waveform, sample_rate)

        try:
            assert temp_path.exists()
            assert temp_path.suffix == ".wav"

            # Load and verify
            import torchaudio

            loaded, sr = torchaudio.load(str(temp_path))
            assert sr == 16000
            assert loaded.shape[0] == 1
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()

    def test_save_temp_wav_different_sample_rates(self):
        """Test saving with different sample rates."""
        for sr in [8000, 16000, 44100]:
            waveform = torch.randn(1, sr)
            temp_path = save_temp_wav(waveform, sr)

            try:
                assert temp_path.exists()
                import torchaudio

                loaded, loaded_sr = torchaudio.load(str(temp_path))
                assert loaded_sr == sr
            finally:
                if temp_path.exists():
                    temp_path.unlink()
