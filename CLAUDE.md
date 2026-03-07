# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local, GPU-accelerated speech transcription pipeline with speaker diarization and LLM post-processing. Targets a single NVIDIA RTX 4070 Laptop (8GB VRAM) — models run serially, one at a time, with explicit unload between stages.

**Language**: Python (GPU-bound inference; orchestration overhead is negligible)

## Architecture

Four-stage serial pipeline, each loading/unloading its own model to stay under 8GB VRAM:

| Stage | Module | Model | VRAM |
|-------|--------|-------|------|
| 0 | `vad.py` + `audio_preprocessing.py` | Silero VAD v5 (CPU) | ~50MB |
| 1 | `diarization.py` + `speaker_registry.py` | pyannote 3.1 + WeSpeaker | ~1-2GB |
| 2 | `transcription.py` | Qwen3-ASR-1.7B (vLLM or transformers) | ~2GB |
| 3 | `llm_postprocess.py` | Qwen3.5-9B Q4_K_M (llama-cpp-python) | ~5.5GB |

Orchestrated by `pipeline.py`. Configuration in `config.yaml`. Output via `output_formatter.py` (JSON, SRT, RTTM, TXT).

## Key Design Constraints

- **VRAM budget**: Peak 8GB per stage. Always `del model` → `gc.collect()` → `torch.cuda.empty_cache()` → `torch.cuda.synchronize()` in that order.
- **Lazy imports**: Heavy deps (`torch`, `torchaudio`, `pyannote.audio`, `qwen_asr`, `llama_cpp`, `vllm`) are imported inside stage functions, not at module level.
- **Crash recovery**: Intermediate results saved to disk (JSON) after each stage. Pipeline can restart from any stage.
- **Chinese-English code-switching**: ASR uses `language=None` for auto-detection. LLM post-processing must preserve code-switching style — never translate between languages.
- **LLM safety**: Stage 3 LLM may only modify speaker labels and punctuation, never alter transcribed words. TPST (Transcript-Preserving Speaker Transfer) check validates this.
- **State passing**: Stages exchange plain Python dicts/lists. Original waveform stays in CPU RAM. Only the active model uses GPU.

## Commands

```bash
# Install dependencies (preferred: uv)
uv sync

# Run transcription
uv run scripts/transcribe.py <audio_file> [--config config.yaml] [--output-dir output/]
uv run scripts/transcribe.py meeting.mp4 --num-speakers 3

# Enroll a speaker voice print
uv run scripts/enroll_speaker.py --name "Alice" --audio sample1.wav sample2.wav

# Run tests
uv run pytest tests/

# Docker
docker run --gpus all -v /path/to/audio:/data pipeline:latest /data/meeting.wav
```

## Dependencies

Defined in `pyproject.toml`. Core: `torch`, `torchaudio`, `silero-vad`, `pyannote.audio`, `qwen-asr[vllm]`, `vllm`, `llama-cpp-python`. Requires Python >=3.10.

pyannote requires accepting the HuggingFace license agreement and setting `HF_TOKEN`.

## Implementation Plan

See `implementation-plan.md` for the full spec, including model references, API usage examples, prompt templates for LLM post-processing, and verification plan.
