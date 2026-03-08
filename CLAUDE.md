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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ASR-Pipeline** (545 symbols, 1298 relationships, 42 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/ASR-Pipeline/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ASR-Pipeline/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ASR-Pipeline/clusters` | All functional areas |
| `gitnexus://repo/ASR-Pipeline/processes` | All execution flows |
| `gitnexus://repo/ASR-Pipeline/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
