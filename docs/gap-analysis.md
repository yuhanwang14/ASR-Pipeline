# Gap Analysis: Pipeline vs Gemini Zero-Shot

Baseline evaluation on `Zoom - Mar 7.m4a` (4 speakers, ~15 min Chinese-English meeting).

## Eval Results (2026-03-08)

| Metric | Pipeline | Notes |
|--------|----------|-------|
| CER | 13.2% | Decent for mixed CN/EN |
| WER | 7.1% | Good overall |
| cpWER | 13.3% | Dragged down by diarization |
| Speaker 1 (→SPEAKER_03) | WER 9.5% | Good |
| Speaker 2 (→SPEAKER_02) | WER 7.5% | Good |
| Speaker 3 (→SPEAKER_01) | WER 69.0% | Merged with Speaker 4 |
| Speaker 4 (→SPEAKER_00) | WER 100.0% | Lost — phantom line only |

Gemini reference has 65 segments across 4 speakers. Pipeline produced 159 segments (over-segmented) and effectively merged two speakers into one.

## Root Causes

### 1. ASR Has No Cross-Segment Context

The pipeline slices audio into ~2s segments (via VAD + diarization), saves each as a temp WAV, and feeds them independently to Qwen3-ASR-1.7B. Each `transcribe()` call sees only its own clip — no surrounding conversation, no topic awareness.

This causes English words in Chinese context to be misrecognized:
- "sync" → "think"
- "VC" → "飞机"
- "Herschel" → "何少"
- "fund raising" → "Fun Reason"
- "SDK" → "SCK"
- "SOTA" → "saota"
- "moat" → "Mot"

Gemini processes the entire audio as one unit — it hears the full business meeting and uses topic context to disambiguate.

### 2. Diarization Fails on Low-Activity Speakers

pyannote 3.1 uses only voice embeddings for speaker assignment. When a speaker talks rarely and briefly (Grace: sparse lines, Stephen: 1 turn), there isn't enough audio to build a distinct voiceprint. Result: they get merged into one speaker.

Gemini can use both voice AND conversational cues ("Grace呢？" → next speaker is Grace) because it jointly models audio and language.

### 3. Error Cascade Across Stages

Pipeline: VAD → Diarization → ASR → LLM postprocess. Each stage's errors feed into the next:
- VAD may cut mid-word
- Diarization assigns wrong speaker
- ASR transcribes a tiny segment without context → more errors
- LLM postprocess only sees text, can't re-listen to audio

Gemini: one model, one pass, no cascade.

### 4. Over-Segmentation

159 segments vs Gemini's 65. The pipeline chops audio too aggressively, producing many empty or near-empty segments. This fragments speaker turns and loses conversational flow.

### 5. Model Scale

Qwen3-ASR 1.7B + Qwen3.5 9B Q4 vs Gemini's hundreds of billions. Not a fair fight on raw capability, but this is an inherent constraint (8GB VRAM, single 4070 Laptop).

## LLM Postprocessing Effectiveness

Stage 3 (Qwen3.5-9B) made 27 changes: 8 text corrections, 19 speaker reassignments. All 8 text corrections were valid — the LLM reasoning approach works. But it's fundamentally a band-aid: it can only fix what's inferable from text patterns, not re-listen to audio.

## Potential Improvements

### Feed larger audio chunks to ASR
Instead of 2s clips, feed 30-60s windows or full speaker turns. Qwen3-ASR can handle longer audio. Trade-off: need to re-align ASR output back to diarization timestamps, and VRAM usage increases.

### Diarization-aware ASR batching
Group consecutive segments from the same speaker turn into one ASR call. Preserves speaker labels while giving the model more context.

### Two-pass approach
1. First pass: ASR on large chunks for better text quality
2. Second pass: align ASR text back to diarization segments for speaker labels

### Accept the trade-off
Local + private + free vs cloud + API cost + better quality. The pipeline exists for the 8GB VRAM constraint. Narrowing the gap is realistic; closing it is not.
