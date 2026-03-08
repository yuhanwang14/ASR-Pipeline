# ASR Pipeline

[![CI](https://github.com/yuhanwang14/ASR-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yuhanwang14/ASR-Pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Local, GPU-accelerated speech transcription pipeline with speaker diarization and LLM-based error correction. Runs entirely on a single consumer NVIDIA GPU (tested on RTX 4070 Laptop, 8 GB VRAM) -- no cloud APIs, no data leaves your machine.

> **Status**: Alpha (v0.1.0) -- functional and tested, but APIs may change.

## Why This Exists

Cloud transcription services are accurate but require uploading private audio. This pipeline gives you speaker-aware transcription that runs locally, optimized for Chinese-English code-switching scenarios (business meetings, tech discussions). It fits four separate models into 8 GB VRAM by loading them one at a time.

## How It Works

The pipeline runs four stages serially. Each stage loads its model, processes the audio, saves intermediate results to disk, then unloads the model to free VRAM before the next stage begins.

```
Audio File
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 0: Voice Activity Detection   │  Silero VAD v5 (CPU, ~50 MB)
│  Removes silence, builds timestamp   │  Outputs: speech segments + timestamp map
│  map for later remapping             │
└──────────────────────────────────────┘
    │ clean waveform (silence removed)
    ▼
┌──────────────────────────────────────┐
│  Stage 1: Speaker Diarization        │  pyannote 3.1 + WeSpeaker (~1-2 GB)
│  "Who spoke when" on clean audio,    │  Outputs: speaker-labeled segments
│  then remaps timestamps to original  │  with original-audio timestamps
└──────────────────────────────────────┘
    │ segments with speaker labels
    ▼
┌──────────────────────────────────────┐
│  Stage 2: Speech-to-Text             │  Qwen3-ASR-1.7B (~2 GB)
│  Transcribes each segment from the   │  Backend: vLLM (fast) or
│  original waveform (not the clean    │  transformers (simpler)
│  waveform — preserves full context)  │
└──────────────────────────────────────┘
    │ segments with text
    ▼
┌──────────────────────────────────────┐
│  Stage 3: LLM Post-Processing        │  Qwen3.5-9B Q4_K_M (~5.5 GB)
│  Fixes speaker label errors and ASR  │  via llama-cpp-python
│  text mistakes using JSON error-pair │
│  output (never rewrites transcript)  │
└──────────────────────────────────────┘
    │
    ▼
  Output: JSON, SRT, RTTM, TXT
```

### VRAM Management

Each stage follows the same pattern: check available VRAM, load model, run inference, `del model` → `gc.collect()` → `torch.cuda.empty_cache()` → `torch.cuda.synchronize()`. A `gpu_stage` context manager enforces this cleanup automatically.

### Crash Recovery

After each stage completes, results are saved as `stage_N.json` in the output directory. If the pipeline crashes (e.g., OOM on Stage 3), re-run with `--resume` to skip already-completed stages.

### LLM Safety

Stage 3 never rewrites the transcript. Instead, it outputs a JSON array of specific corrections:

- **Speaker corrections**: `{"line": 1, "old_speaker": "SPEAKER_00", "new_speaker": "SPEAKER_01"}`
- **Text corrections**: `{"line": 0, "original": "think", "corrected": "sync"}`

Each correction is validated individually before being applied. A hallucination guard rejects all corrections if the LLM produces too many. A TPST (Transcript-Preserving Speaker Transfer) check provides an additional safety layer by comparing word tokens before and after processing.

## Features

- **Chinese-English code-switching** -- ASR auto-detects language; LLM post-processing fixes common cross-language ASR errors (e.g., "VC" misheard as "飞机", "SDK" → "SCK")
- **Speaker identification** -- pyannote diarization with optional voice-print enrollment for known speakers (WeSpeaker cosine similarity matching)
- **Crash recovery** -- intermediate JSON saved after each stage; resume from the last completed stage with `--resume`
- **LLM error correction** -- speaker label correction and text error correction via quantized Qwen3.5-9B with per-patch validation
- **Multiple output formats** -- JSON (full metadata), SRT (subtitles), RTTM (diarization eval), TXT (plain text with timestamps)
- **Evaluation toolkit** -- built-in CER, WER, and cpWER scoring against reference transcripts
- **Protocol-based backends** -- all GPU stages use Python Protocol classes for dependency injection, making every module testable without hardware

## Supported Audio Formats

WAV, FLAC, OGG (via soundfile), and M4A, MP4, WebM, and other formats (via ffmpeg fallback). Audio is automatically converted to 16 kHz mono.

## Requirements

- **Python** >= 3.10
- **NVIDIA GPU** with CUDA support (8 GB VRAM recommended)
- **ffmpeg** (optional, for M4A/MP4/WebM input)
- **HuggingFace token** for pyannote models -- accept the [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) license, then set `HF_TOKEN` or `ASR_HF_TOKEN` as an environment variable
- **GGUF model file** for Stage 3 -- download Qwen3.5-9B Q4_K_M from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) and place it at the path configured in `config.yaml`

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/yuhanwang14/ASR-Pipeline.git && cd ASR-Pipeline
uv sync                # install all dependencies
uv sync --group dev    # include pytest, ruff, pytest-cov
```

### Using pip

```bash
git clone https://github.com/yuhanwang14/ASR-Pipeline.git && cd ASR-Pipeline
pip install -e .
pip install -e ".[dev]"   # includes pytest, ruff
```

### Docker

```bash
docker build -t asr-pipeline .
docker run --gpus all -v /path/to/audio:/data asr-pipeline /data/meeting.wav
```

The Docker image is based on `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and uses `uv` for dependency management.

## Usage

### Transcribe audio

```bash
uv run scripts/transcribe.py recording.wav
uv run scripts/transcribe.py meeting.mp4 --num-speakers 3
uv run scripts/transcribe.py call.wav --output-dir results/ --formats json srt txt
uv run scripts/transcribe.py lecture.wav --verbose
```

Resume after a crash:

```bash
uv run scripts/transcribe.py meeting.wav --resume
```

### CLI options

| Flag | Description |
|------|-------------|
| `--config FILE` | Path to config file (default: `config.yaml`) |
| `--output-dir DIR` | Output directory (default: `output/`) |
| `--num-speakers N` | Number of speakers (auto-detect if not set) |
| `--formats FMT...` | Output formats: `json`, `srt`, `rttm`, `txt` |
| `--resume` | Resume from last completed stage |
| `--verbose`, `-v` | Enable verbose logging |

### Enroll speaker voice prints

Pre-enroll known speakers so the pipeline can label them by name instead of `SPEAKER_00`:

```bash
uv run scripts/enroll_speaker.py --name "Alice" --audio sample1.wav sample2.wav
uv run scripts/enroll_speaker.py --list
uv run scripts/enroll_speaker.py --name "Alice" --delete
```

Voice prints are stored as `.npy` files in `speaker_profiles/` and matched via cosine similarity against WeSpeaker embeddings during diarization.

### Evaluate output quality

Compare pipeline output against a reference transcript:

```bash
# Evaluate pre-computed output
uv run scripts/evaluate.py --output output/meeting.json --reference reference.md

# Run pipeline and evaluate in one step
uv run scripts/evaluate.py meeting.m4a --reference reference.md
```

Produces a scorecard with CER (Character Error Rate), WER (Word Error Rate), and cpWER (concatenated minimum-permutation Word Error Rate, a speaker-aware metric).

### Output formats

| Format | Extension | Content |
|--------|-----------|---------|
| JSON | `.json` | Full metadata, speaker list, timestamped segments |
| SRT | `.srt` | Subtitles with `[Speaker]` prefix per line |
| RTTM | `.rttm` | Rich Transcription Time Marked (standard diarization eval format) |
| TXT | `.txt` | `[HH:MM:SS] Speaker: text` per line |

## Configuration

All settings live in [`config.yaml`](config.yaml). Key sections:

```yaml
vad:
  merge_gap_seconds: 0.5        # merge speech segments closer than this

diarization:
  model: "pyannote/speaker-diarization-3.1"
  num_speakers: null             # null = auto-detect
  match_threshold: 0.75          # cosine similarity for enrolled speaker matching

asr:
  model: "Qwen/Qwen3-ASR-1.7B"
  backend: "transformers"        # "vllm" (faster) or "transformers" (simpler setup)
  language: null                 # null = auto-detect (best for code-switching)

llm:
  model_path: "models/Qwen3.5-9B-Q4_K_M.gguf"
  n_ctx: 16384                   # context window size
  n_gpu_layers: -1               # -1 = all layers on GPU
  tasks:
    speaker_correction: true
    text_correction: true

output:
  formats: ["json", "srt", "txt"]
  save_intermediate: true        # enable crash recovery
```

## Known Limitations

This is a local pipeline on consumer hardware, not a cloud service. Understand the trade-offs:

- **Code-switching errors** -- each audio segment is transcribed independently without cross-segment context, so English words in Chinese speech are sometimes misrecognized (e.g., "sync" → "think", "VC" → "飞机"). The LLM post-processing catches some of these, but not all.
- **Over-segmentation** -- the VAD + diarization pipeline can produce many short segments, fragmenting speaker turns.
- **Low-activity speakers** -- speakers with very few or very short utterances may be merged with another speaker by the diarization model, since there isn't enough audio to build a distinct voiceprint.
- **VRAM ceiling** -- peak usage is ~5.5 GB (Stage 3). All four models cannot run simultaneously.

For a detailed comparison against cloud baselines, see [`docs/gap-analysis.md`](docs/gap-analysis.md).

## Development

### Running tests

```bash
uv run pytest tests/                           # unit tests (no GPU needed)
uv run pytest tests/ -v --tb=short             # verbose output
uv run pytest tests/ -k "not integration and not gpu"  # CI-safe subset
uv run pytest tests/ -m gpu                    # GPU integration tests only
```

The test suite uses Protocol-based dependency injection -- all GPU backends have mock implementations so unit tests run without CUDA.

### Linting and formatting

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

### Project structure

```
ASR-Pipeline/
├── config.yaml               # Pipeline configuration
├── pyproject.toml             # Dependencies and tool config
├── Dockerfile                 # GPU-enabled container (CUDA 12.4)
├── scripts/
│   ├── transcribe.py          # Main CLI entry point
│   ├── enroll_speaker.py      # Speaker enrollment CLI
│   └── evaluate.py            # Evaluation scorecard (CER/WER/cpWER)
├── src/
│   ├── pipeline.py            # Orchestrator (stage sequencing, crash recovery)
│   ├── audio_preprocessing.py # Audio loading, resampling, slicing
│   ├── vad.py                 # Stage 0: Silero VAD
│   ├── diarization.py         # Stage 1: pyannote speaker diarization
│   ├── speaker_registry.py    # Voice print enrollment and matching
│   ├── transcription.py       # Stage 2: Qwen3-ASR (vLLM / transformers)
│   ├── llm_postprocess.py     # Stage 3: LLM error correction
│   ├── timestamp_utils.py     # VAD timestamp remapping
│   ├── eval_metrics.py        # CER, WER, cpWER computation
│   ├── output_formatter.py    # JSON, SRT, RTTM, TXT formatters
│   ├── gpu_utils.py           # VRAM checking and cleanup
│   ├── intermediate.py        # Stage result serialization
│   ├── config.py              # YAML config loader
│   └── logging_config.py      # Colored, stage-aware logging
├── tests/
│   └── test_*.py              # 14 test modules, 126+ tests
├── speaker_profiles/          # Enrolled voice prints (.npy)
└── output/                    # Default output directory
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting issues or pull requests.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Security

To report a vulnerability, please see [SECURITY.md](SECURITY.md). Do **not** open a public issue for security bugs.

## License

MIT -- see [LICENSE](LICENSE) for details.
