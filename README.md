# ASR Pipeline

Local, GPU-accelerated speech transcription pipeline with speaker diarization and LLM post-processing. Designed for a single NVIDIA GPU (tested on RTX 4070 Laptop, 8 GB VRAM). Models run serially -- each stage loads its model, runs inference, and unloads before the next stage begins.

## Architecture

Four-stage serial pipeline, each staying within the 8 GB VRAM budget:

| Stage | Module | Model | VRAM |
|-------|--------|-------|------|
| 0 -- VAD | `vad.py` + `audio_preprocessing.py` | Silero VAD v5 (CPU) | ~50 MB |
| 1 -- Diarization | `diarization.py` + `speaker_registry.py` | pyannote 3.1 + WeSpeaker | ~1-2 GB |
| 2 -- ASR | `transcription.py` | Qwen3-ASR-1.7B (vLLM or transformers) | ~2 GB |
| 3 -- LLM Post-processing | `llm_postprocess.py` | Qwen3.5-9B Q4_K_M (llama-cpp-python) | ~5.5 GB |

The pipeline is orchestrated by `pipeline.py`. Configuration lives in `config.yaml`.

## Features

- **Chinese-English code-switching** -- ASR auto-detects language per segment; LLM post-processing preserves the original language mix.
- **Speaker identification** -- pyannote diarization with optional voice-print enrollment for known speakers (WeSpeaker embeddings).
- **Crash recovery** -- intermediate results are saved to disk as JSON after each stage; the pipeline can resume from the last completed stage.
- **LLM post-processing** -- speaker label correction, text error correction, and meeting summarization via a quantized LLM with TPST safety checks.
- **Multiple output formats** -- JSON, SRT, RTTM, and plain text.

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (8 GB VRAM recommended)
- HuggingFace token for pyannote models (accept the license at huggingface.co, then set `HF_TOKEN`)

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone <repo-url> && cd ASR-Pipeline

# Install all dependencies
uv sync

# Install with dev tools (pytest, ruff)
uv sync --group dev
```

### Using pip

```bash
pip install -e .
pip install -e ".[dev]"   # includes pytest, ruff
```

### Docker

```bash
docker build -t asr-pipeline .
docker run --gpus all -v /path/to/audio:/data asr-pipeline /data/meeting.wav
```

## Usage

### Transcribe audio

```bash
uv run scripts/transcribe.py recording.wav
uv run scripts/transcribe.py meeting.mp4 --num-speakers 3
uv run scripts/transcribe.py call.wav --output-dir results/ --formats json srt txt
uv run scripts/transcribe.py lecture.wav --no-summary --verbose
```

Resume after a crash (picks up from the last completed stage):

```bash
uv run scripts/transcribe.py meeting.wav --resume
```

### Enroll a speaker voice print

```bash
uv run scripts/enroll_speaker.py --name "Alice" --audio sample1.wav sample2.wav
uv run scripts/enroll_speaker.py --list
uv run scripts/enroll_speaker.py --name "Alice" --delete
```

### Output formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON   | `.json`   | Full metadata, speakers, segments, and summary |
| SRT    | `.srt`    | Subtitles with speaker tags |
| RTTM   | `.rttm`   | Rich Transcription Time Marked (for evaluation) |
| TXT    | `.txt`    | Plain text with timestamps |

## Configuration

All settings are in `config.yaml`. Key sections:

- `vad` -- Silero VAD thresholds and merge gap
- `diarization` -- pyannote model, HF token, number of speakers
- `asr` -- Qwen3-ASR model, backend (vLLM or transformers), language
- `llm` -- GGUF model path, context length, post-processing tasks
- `output` -- formats and output directory

See `config.yaml` for the full reference with comments.

## Development

### Running tests

```bash
uv run pytest tests/
uv run pytest tests/ -v --tb=short
```

### Linting and formatting

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Project structure

```
ASR-Pipeline/
  config.yaml            # Pipeline configuration
  pyproject.toml         # Dependencies and tool config
  scripts/
    transcribe.py        # CLI entry point
    enroll_speaker.py    # Speaker enrollment CLI
  src/
    pipeline.py          # Orchestrator
    audio_preprocessing.py
    vad.py               # Stage 0: Voice Activity Detection
    diarization.py       # Stage 1: Speaker Diarization
    speaker_registry.py  # Voice print enrollment and matching
    transcription.py     # Stage 2: ASR
    llm_postprocess.py   # Stage 3: LLM Post-processing
    output_formatter.py  # JSON, SRT, RTTM, TXT output
    gpu_utils.py         # VRAM management
    intermediate.py      # Crash recovery serialization
    config.py            # Config loader
    logging_config.py    # Logging setup
  tests/
    test_*.py            # Unit and integration tests
  speaker_profiles/      # Enrolled voice prints (.npy)
  output/                # Default output directory
```

## License

MIT -- see [LICENSE](LICENSE) for details.
