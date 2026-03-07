# Implementation Plan: Local Speech Transcription Pipeline with Speaker Diarization and LLM Post-Processing

## Context

Build a fully local, GPU-accelerated speech transcription pipeline that:
- Identifies **who is speaking** (speaker diarization + voice print recognition)
- Transcribes **Chinese-English code-switched** speech with SOTA accuracy
- Uses an **LLM for post-processing** (error correction, speaker label refinement, summarization)
- Runs serially on a single **NVIDIA RTX 4070 Laptop (8GB VRAM)** + 31GB RAM + Intel i9-13900H
- Processes audio files (not real-time streaming)

### Hardware Constraints
- **Peak VRAM per stage**: must stay under 8GB
- **Strategy**: serial execution — load one model at a time, unload before loading the next
- **RAM**: 31GB available for model loading/unloading overhead and audio data

### Models Used (Serial, Not Concurrent)

| Stage | Model | VRAM Usage | Role |
|-------|-------|------------|------|
| 0 | [Silero VAD v5](https://github.com/snakers4/silero-vad) | ~50MB | Voice Activity Detection — remove silence |
| 1 | [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) + [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ~1-2GB | Speaker diarization + speaker embeddings |
| 2 | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ~2GB | SOTA speech-to-text (Chinese-English code-switching) |
| 3 | [Qwen3.5-9B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | ~5.5GB | LLM post-processing: error correction, speaker label refinement, summarization |

---

## Language Evaluation Summary

### Why Python?

The pipeline is **GPU-bound** — actual inference runs in compiled CUDA kernels regardless of the orchestration language. Python adds ~1-5% overhead for glue code. For a 1-hour audio file, GPU inference takes ~5-15 minutes; Python orchestration adds ~5-15 seconds.

| Criteria | Python | Rust | C++ | Go |
|----------|--------|------|-----|-----|
| All models supported | Yes | **No** (Qwen3-ASR blocker) | Yes | No |
| GPU inference speed | Same | Same | Same | N/A |
| Orchestration overhead | ~1-5% | ~0.1% | ~0.1% | ~0.5% |
| Startup time | 3-5s | <0.5s | <0.5s | <0.5s |
| RAM overhead | ~800MB-1GB | ~50-100MB | ~50-100MB | ~100-200MB |
| Development time | 2-4 weeks | 6-12 weeks | 8-16 weeks | Not viable |
| Single binary distribution | No | Yes | Yes | Yes |
| ML ecosystem maturity | Best | Growing | Good | Poor |

### Why not the alternatives?

- **Rust**: Qwen3-ASR-1.7B has no ONNX export and no native Rust implementation without libtorch (~500MB C++ dependency). Would require switching to Whisper (worse at Chinese-English code-switching). Deepgram's 30-80% Rust speedup is for high-throughput serving with thousands of concurrent streams, not single-file offline processing.
- **C++**: Only alternative with a complete ecosystem (antirez/qwen-asr exists for Qwen3-ASR in pure C). However, full C++ pipeline = months of development (CMake builds, manual memory management, dependency wrangling). Best suited if building a distributable product, not a personal tool.
- **Go**: No speaker diarization or ASR support exists. Would wrap C/C++ via cgo for 3+ pipeline stages, defeating the purpose. GC pauses (1-10ms) unsuitable for audio processing.

### Where real speedups come from (not language choice)

1. **vLLM backend for Qwen3-ASR** — batch inference, continuous batching, PagedAttention (2-5x)
2. **Flash Attention** across all stages (2-4x attention speedup, less VRAM)
3. **Speculative decoding** for Stage 3 LLM (2-3x token generation)
4. **Batch short audio segments** in Stage 2 instead of one at a time
5. **Overlap CPU preprocessing with GPU inference** via Python asyncio/threading

Evidence: vLLM (Python) outperforms llama.cpp (C++) for batched inference through algorithmic optimization. PyTorch Python vs C++ (libtorch) benchmarks show execution times are "very close" for large models.

---

## Project Structure

```
transcription-pipeline/
├── README.md
├── pyproject.toml                  # Project config, dependencies
├── config.yaml                     # Runtime configuration (paths, model IDs, thresholds)
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # Main orchestrator — runs all stages serially
│   ├── gpu_utils.py                # GPU memory management utilities
│   ├── audio_preprocessing.py      # Audio loading, resampling, format conversion
│   ├── vad.py                      # Stage 0: Silero VAD
│   ├── diarization.py              # Stage 1: pyannote + WeSpeaker
│   ├── transcription.py            # Stage 2: Qwen3-ASR
│   ├── llm_postprocess.py          # Stage 3: Qwen3.5 LLM post-processing
│   ├── speaker_registry.py         # Known speaker enrollment and matching
│   └── output_formatter.py         # Format final output (JSON, SRT, TXT, RTTM)
├── speaker_profiles/               # Stored speaker voice prints (numpy embeddings)
│   └── .gitkeep
├── tests/
│   ├── test_pipeline.py
│   ├── test_audio_preprocessing.py
│   └── test_fixtures/
│       └── sample_audio.wav        # Short test audio file
└── scripts/
    ├── enroll_speaker.py           # CLI tool to register a known speaker
    └── transcribe.py               # CLI entry point
```

---

## Stage 0: Audio Preprocessing + VAD

### File: `src/audio_preprocessing.py`

**Purpose**: Load any audio/video file, convert to 16kHz mono WAV, normalize.

**Implementation**:

1. Load audio using [torchaudio](https://pytorch.org/audio/stable/index.html) (`torchaudio.load()`)
   - Supports WAV, FLAC, MP3, OGG, Opus via ffmpeg backend
   - Ref: [torchaudio.load docs](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.load.html)
2. Resample to 16kHz using `torchaudio.transforms.Resample(orig_freq, 16000)`
   - Ref: [torchaudio.transforms.Resample](https://docs.pytorch.org/audio/stable/transforms.html)
3. Convert to mono: `waveform.mean(dim=0, keepdim=True)`
4. Normalize amplitude to [-1, 1]
5. Return `(waveform: torch.Tensor, sample_rate: int)`

**Input**: any audio/video file path
**Output**: `torch.Tensor` (1 x N) at 16kHz mono

### File: `src/vad.py`

**Purpose**: Remove silence segments to improve diarization accuracy and reduce processing time.

**Implementation**:

1. Load [Silero VAD v5](https://github.com/snakers4/silero-vad) model:
   ```python
   from silero_vad import load_silero_vad, get_speech_timestamps
   model = load_silero_vad()
   ```
   - Ref: [Silero VAD GitHub](https://github.com/snakers4/silero-vad), [PyTorch Hub](https://pytorch.org/hub/snakers4_silero-vad_vad/)
   - Runs on CPU, <1ms per audio chunk, negligible memory
2. Get speech timestamps:
   ```python
   speech_timestamps = get_speech_timestamps(waveform, model, return_seconds=True)
   ```
3. Return list of `{start: float, end: float}` segments containing speech
4. Optionally merge segments with gaps < 0.5s to avoid over-fragmentation
5. Concatenate speech segments into a clean waveform for diarization, keeping a **timestamp mapping** to reconstruct original positions later

**Input**: `torch.Tensor` (16kHz mono)
**Output**: `(clean_waveform: torch.Tensor, timestamp_map: List[dict])`

---

## Stage 1: Speaker Diarization

### File: `src/diarization.py`

**Purpose**: Determine who speaks when. Output time-stamped speaker segments.

**Implementation**:

1. Load [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) pipeline:
   ```python
   from pyannote.audio import Pipeline
   pipeline = Pipeline.from_pretrained(
       "pyannote/speaker-diarization-3.1",
       use_auth_token="YOUR_HF_TOKEN"  # Required — accept license on HuggingFace first
   )
   pipeline.to(torch.device("cuda"))
   ```
   - Ref: [pyannote-audio GitHub](https://github.com/pyannote/pyannote-audio)
   - Ref: [pyannote speaker-diarization-3.1 model card](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - **NOTE**: pyannote 3.1 requires accepting user agreement on HuggingFace and generating a token
   - Internally uses [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) for speaker embeddings (EER 0.72% on VoxCeleb1-O)
   - DER (Diarization Error Rate): ~10-11% on standard benchmarks
   - Real-time factor: ~2.5% on GPU (1hr audio ≈ 90s processing)

2. Run diarization:
   ```python
   diarization = pipeline({"waveform": waveform, "sample_rate": 16000})
   # Optional: specify num_speakers if known
   # diarization = pipeline(audio, num_speakers=3)
   ```

3. Extract segments — the output is a [pyannote.core.Annotation](https://pyannote.github.io/pyannote-core/structure.html#annotation) object:
   ```python
   segments = []
   for turn, _, speaker in diarization.itertracks(yield_label=True):
       segments.append({
           "start": turn.start,    # float, seconds
           "end": turn.end,        # float, seconds
           "speaker": speaker      # str, e.g. "SPEAKER_00"
       })
   ```

4. Map timestamps back to original audio positions using the VAD timestamp_map from Stage 0

5. Extract per-speaker embeddings for known-speaker matching (see `speaker_registry.py`)

6. **Unload model**:
   ```python
   del pipeline
   gc.collect()
   torch.cuda.empty_cache()
   ```

**Input**: `(waveform: torch.Tensor, sample_rate: int)`
**Output**: `List[DiarizationSegment]` where each segment has `{start, end, speaker, embedding}`

### File: `src/speaker_registry.py`

**Purpose**: Register known speakers by voice print. Match anonymous diarization labels (SPEAKER_00, SPEAKER_01) to real names.

**Implementation**:

1. Use [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) via pyannote's `Inference` API for speaker embedding extraction:
   ```python
   from pyannote.audio import Model, Inference
   model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
   inference = Inference(model, window="whole")
   inference.to(torch.device("cuda"))
   embedding = inference("enrollment_audio.wav")  # Returns (1, D) numpy array
   ```
   - Ref: [WeSpeaker GitHub](https://github.com/wenet-e2e/wespeaker)
   - Ref: [WeSpeaker ResNet34-LM model card](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
   - Trained on [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (5994 speakers)

2. **Enrollment** (one-time per speaker):
   - Accept one or more audio clips of the speaker
   - Extract embeddings, average them
   - Save to `speaker_profiles/{name}.npy`

3. **Matching** (during pipeline):
   - For each diarized speaker cluster, extract representative embedding
   - Compute cosine similarity against all enrolled profiles
   - If similarity > threshold (default 0.75), assign real name
   - Otherwise keep anonymous label (SPEAKER_XX)

4. Run speaker matching as part of Stage 1 (while WeSpeaker model is still loaded), before unloading

**Input**: diarization segments + speaker_profiles directory
**Output**: updated segments with `speaker` field replaced by real names where matched

---

## Stage 2: Speech-to-Text Transcription

### File: `src/transcription.py`

**Purpose**: Transcribe each speaker segment using Qwen3-ASR-1.7B, the current open-source SOTA for Chinese-English code-switching.

**Implementation**:

1. Load [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model via **vLLM backend** (recommended — 2-5x faster with continuous batching and PagedAttention):
   ```python
   from qwen_asr import Qwen3ASRModel
   model = Qwen3ASRModel.LLM(
       model="Qwen/Qwen3-ASR-1.7B",
       gpu_memory_utilization=0.7,
       max_new_tokens=4096,
   )
   ```
   - Ref: [vLLM Qwen3-ASR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html)
   - Ref: [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
   - Ref: [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)
   - Ref: [qwen-asr PyPI](https://pypi.org/project/qwen-asr/)
   - Accuracy: Chinese ~98%, English ~97%, code-switching SOTA
   - Supports 52 languages and dialects
   - Trained on Qwen3-Omni foundation, distilled for ASR
   - Install: `pip install -U qwen-asr[vllm]`

2. **Fallback** — direct transformers backend (simpler setup, no vLLM dependency):
   ```python
   model = Qwen3ASRModel.from_pretrained(
       "Qwen/Qwen3-ASR-1.7B",
       dtype=torch.bfloat16,
       device_map="cuda:0",
       max_new_tokens=4096,
   )
   ```
   - Install: `pip install qwen-asr`
   - Use when vLLM installation fails or for simpler debugging

3. For each diarization segment:
   - Slice the original audio waveform using the segment's `start` and `end` timestamps:
     ```python
     start_sample = int(segment["start"] * sample_rate)
     end_sample = int(segment["end"] * sample_rate)
     segment_waveform = waveform[:, start_sample:end_sample]
     ```
   - Save segment to a temporary WAV file (Qwen3-ASR accepts file paths)
   - Transcribe:
     ```python
     results = model.transcribe(audio=temp_wav_path, language=None)
     text = results[0].text
     ```
   - `language=None` enables auto-detection (important for code-switching)

4. **Batching strategy**: group short segments (< 5s) and process them together for efficiency. Long segments (> 5 min) should be split at silence boundaries.

5. **Unload model**:
   ```python
   del model
   gc.collect()
   torch.cuda.empty_cache()
   ```

**Input**: `(waveform: torch.Tensor, segments: List[DiarizationSegment])`
**Output**: `List[TranscriptSegment]` where each has `{start, end, speaker, text}`

---

## Stage 3: LLM Post-Processing

### File: `src/llm_postprocess.py`

**Purpose**: Use Qwen3.5-9B to refine transcription output — correct ASR errors, fix speaker labels based on semantic context, and optionally generate summaries.

**Background**: Research shows LLM post-processing can reduce [WDER](https://arxiv.org/abs/2401.03506) (Word Diarization Error Rate) by up to 55.5%. However, **zero-shot LLM correction often degrades quality** due to hallucination. The approach below uses structured prompting inspired by [DiarizationLM](https://github.com/google/speaker-id/tree/master/DiarizationLM) (Google, INTERSPEECH 2024) and its [TPST (Transcript-Preserving Speaker Transfer)](https://www.isca-archive.org/interspeech_2024/wang24h_interspeech.pdf) algorithm.

**Key Principle**: The LLM must ONLY modify speaker labels and punctuation, NEVER alter the transcribed words themselves. This prevents hallucination.

**Implementation**:

1. Load [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) in Q4_K_M quantization:

   **Option A — llama-cpp-python** (recommended for GGUF, precise VRAM control):
   ```python
   from llama_cpp import Llama
   llm = Llama(
       model_path="path/to/qwen3.5-9b-q4_k_m.gguf",
       n_gpu_layers=-1,  # All layers on GPU
       n_ctx=8192,
   )
   ```
   - Ref: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
   - GGUF models: [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) or [bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF)
   - ~5.5GB VRAM for Q4_K_M

   **Option B — transformers + bitsandbytes**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   quantization_config = BitsAndBytesConfig(load_in_4bit=True)
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen3.5-9B",
       quantization_config=quantization_config,
       device_map="auto",
   )
   ```
   - Ref: [Qwen3.5 Unsloth Guide](https://unsloth.ai/docs/models/qwen3.5)
   - Ref: [Qwen3.5 vLLM Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

2. **Task 1: Speaker Label Correction** (inspired by [DiarizationLM](https://arxiv.org/abs/2401.03506))

   Format the transcript in DiarizationLM's compact text format:
   ```
   <speaker:SPEAKER_00> 今天我们来讨论一下 project timeline
   <speaker:SPEAKER_01> OK let me pull up the schedule
   <speaker:SPEAKER_00> 我觉得 deadline 可以往后推一周
   ```

   Prompt template:
   ```
   You are a transcript editor. Below is a speaker-diarized transcript that may have
   speaker attribution errors. Based on conversational context and semantic coherence,
   correct the speaker labels ONLY. Do NOT modify any words in the transcript.

   Rules:
   1. Only change <speaker:XX> tags
   2. Never add, remove, or modify any spoken words
   3. Use conversational flow to determine correct speaker assignments
   4. If unsure, keep the original label

   Transcript:
   {formatted_transcript}

   Output the corrected transcript in the same format:
   ```

3. **Task 2: Transcription Error Correction**

   Prompt template:
   ```
   You are a transcript proofreader for Chinese-English code-switched speech.
   Fix obvious ASR errors (homophones, misheard words) while preserving the
   original meaning. Fix punctuation. Do NOT change speaker labels.

   Rules:
   1. Fix clear errors only — do not rephrase or paraphrase
   2. Preserve code-switching style (don't translate Chinese to English or vice versa)
   3. Fix punctuation and sentence boundaries
   4. Keep all speaker labels unchanged

   Transcript:
   {corrected_transcript_from_task1}

   Output the corrected transcript:
   ```

4. **Task 3: Summary Generation** (optional, triggered by config flag)

   Prompt template:
   ```
   Summarize the following meeting transcript. Include:
   1. Key discussion points
   2. Decisions made
   3. Action items with assigned speakers
   Output in the same language mix as the original transcript.

   Transcript:
   {final_transcript}
   ```

5. **TPST Safety Check**: After LLM processing, verify that no words were added or removed by comparing word lists (excluding speaker tags and punctuation) between input and output. If words differ, fall back to the original transcript with a warning.
   - Ref: [TPST algorithm — DiarizationLM paper Section 3.2](https://arxiv.org/html/2401.03506v5)

6. **Unload model**:
   ```python
   del llm
   gc.collect()
   torch.cuda.empty_cache()
   ```

**Input**: `List[TranscriptSegment]`
**Output**: `CorrectedTranscript` with `{segments, summary, warnings}`

---

## GPU Memory Management

### File: `src/gpu_utils.py`

**Purpose**: Ensure clean model loading/unloading between stages to stay within 8GB VRAM.

**Implementation**:

```python
import gc
import torch

def unload_model(*models):
    """Forcefully unload PyTorch models from GPU."""
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_vram_usage():
    """Return current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def check_vram_available(required_mb: float):
    """Check if enough VRAM is available, raise if not."""
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated()) / 1024 / 1024
        if free < required_mb:
            raise RuntimeError(
                f"Insufficient VRAM: {free:.0f}MB available, {required_mb:.0f}MB required"
            )
```

**Key Notes**:
- PyTorch retains ~254MB CUDA context that cannot be freed — Ref: [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- Always `del model` → `gc.collect()` → `torch.cuda.empty_cache()` in that order
- Log VRAM usage before and after each stage for debugging
- Add a `torch.cuda.synchronize()` call to ensure all GPU ops complete before measuring

---

## Pipeline Orchestrator

### File: `src/pipeline.py`

**Purpose**: Run all stages serially, managing model loading/unloading.

**Implementation**:

```python
def run_pipeline(audio_path: str, config: dict) -> dict:
    """
    Main pipeline entry point.

    Flow:
    1. Preprocess audio → 16kHz mono waveform
    2. VAD → speech segments (CPU, negligible memory)
    3. Load pyannote + WeSpeaker → diarization + speaker matching → unload
    4. Load Qwen3-ASR → transcribe each segment → unload
    5. Load Qwen3.5-9B → post-process → unload
    6. Format and save output
    """
```

**State Passing Between Stages**:
- Each stage receives and returns plain Python data structures (dicts, lists)
- The original waveform tensor stays in CPU RAM throughout
- Only the active model occupies GPU VRAM
- Intermediate results are saved to disk (JSON) after each stage for crash recovery

**Error Handling**:
- If any stage fails, save intermediate results and allow restart from that stage
- VRAM OOM errors should trigger automatic model unload + retry with lower batch size

---

## Output Formats

### File: `src/output_formatter.py`

**Purpose**: Generate output in multiple standard formats.

**Formats**:

1. **JSON** (primary, machine-readable):
   ```json
   {
     "metadata": {
       "audio_file": "meeting.wav",
       "duration_seconds": 3600,
       "num_speakers": 3,
       "processing_time_seconds": 420
     },
     "speakers": [
       {"id": "SPEAKER_00", "name": "Alice", "confidence": 0.92},
       {"id": "SPEAKER_01", "name": null, "confidence": null}
     ],
     "segments": [
       {
         "start": 0.5,
         "end": 3.2,
         "speaker": "Alice",
         "text": "今天我们来讨论一下 project timeline"
       }
     ],
     "summary": "..."
   }
   ```

2. **SRT** (subtitle format, for video players):
   ```
   1
   00:00:00,500 --> 00:00:03,200
   [Alice] 今天我们来讨论一下 project timeline
   ```

3. **RTTM** (standard diarization format):
   ```
   SPEAKER meeting 1 0.500 2.700 <NA> <NA> Alice <NA> <NA>
   ```
   - Ref: [RTTM Format Specification](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)

4. **Plain Text** (human-readable):
   ```
   [00:00:00] Alice: 今天我们来讨论一下 project timeline
   [00:00:03] SPEAKER_01: OK let me pull up the schedule
   ```

---

## Configuration

### File: `config.yaml`

```yaml
# Audio preprocessing
audio:
  target_sample_rate: 16000
  target_channels: 1  # mono

# VAD settings
vad:
  model: "silero_vad"  # https://github.com/snakers4/silero-vad
  min_speech_duration_ms: 250
  min_silence_duration_ms: 500
  merge_gap_seconds: 0.5

# Speaker diarization
diarization:
  model: "pyannote/speaker-diarization-3.1"  # https://huggingface.co/pyannote/speaker-diarization-3.1
  embedding_model: "pyannote/wespeaker-voxceleb-resnet34-LM"  # https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
  hf_token: null  # Set via env var HF_TOKEN or here
  num_speakers: null  # null = auto-detect, or set integer
  speaker_profiles_dir: "speaker_profiles/"
  match_threshold: 0.75  # Cosine similarity threshold for known speaker matching

# ASR
asr:
  model: "Qwen/Qwen3-ASR-1.7B"  # https://huggingface.co/Qwen/Qwen3-ASR-1.7B
  backend: "vllm"  # "vllm" (recommended) or "transformers" (fallback)
  dtype: "bfloat16"
  max_new_tokens: 4096
  language: null  # null = auto-detect (best for code-switching)
  max_segment_duration: 300  # seconds, split longer segments
  gpu_memory_utilization: 0.7  # vLLM memory allocation fraction
  flash_attention: true  # Enable Flash Attention 2 for faster inference and lower VRAM

# LLM post-processing
llm:
  model_path: null  # Path to GGUF file, or HuggingFace model ID
  model_id: "Qwen/Qwen3.5-9B"  # https://huggingface.co/Qwen/Qwen3.5-9B
  gguf_url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF"  # Download source
  backend: "llama-cpp"  # "llama-cpp" or "transformers"
  n_ctx: 8192
  n_gpu_layers: -1  # -1 = all layers on GPU
  tasks:
    speaker_correction: true
    text_correction: true
    summarization: true

# Output
output:
  formats: ["json", "srt", "txt"]  # Options: json, srt, rttm, txt
  output_dir: "output/"
  save_intermediate: true  # Save results after each stage for crash recovery
```

---

## Performance Optimization

### Lazy Imports

Import heavy dependencies only when their stage runs. This reduces startup time by 2-3s and saves ~200MB RAM for stages that don't execute.

```python
# In pipeline.py — lazy import pattern
def run_stage_2(waveform, segments, config):
    from qwen_asr import Qwen3ASRModel  # Only imported when Stage 2 runs
    model = Qwen3ASRModel.LLM(...)
    ...
```

Apply to: `torch`, `torchaudio`, `pyannote.audio`, `qwen_asr`, `llama_cpp`, `vllm`. Keep only `yaml`, `pathlib`, `json` as top-level imports in `pipeline.py`.

### Per-Stage Timing and Profiling

Add timing instrumentation to `pipeline.py` for every stage:

```python
import time

def run_pipeline(audio_path, config):
    timings = {}
    for stage_name, stage_fn in stages:
        vram_before = get_vram_usage()
        t0 = time.perf_counter()
        result = stage_fn(...)
        timings[stage_name] = time.perf_counter() - t0
        vram_after = get_vram_usage()
        logger.info(f"[{stage_name}] {timings[stage_name]:.1f}s | VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB")
    # Include timings in output metadata
```

This identifies actual bottlenecks rather than guessing. Log output example:
```
[Stage 0: VAD]          1.2s | VRAM: 0MB -> 50MB
[Stage 1: Diarization]  42.3s | VRAM: 50MB -> 1800MB
[Stage 2: ASR]          187.5s | VRAM: 0MB -> 2100MB
[Stage 3: LLM]          95.8s | VRAM: 0MB -> 5500MB
```

### Flash Attention

Enable Flash Attention 2 across all PyTorch-based stages for 2-4x faster attention computation and lower VRAM usage:

- **Stage 1 (pyannote)**: Supported via `torch.backends.cuda.flash_sdp_enabled()`
- **Stage 2 (Qwen3-ASR via vLLM)**: vLLM enables Flash Attention automatically when available
- **Stage 3 (llama.cpp)**: Uses its own Flash Attention implementation, enabled by default in recent versions

Prerequisite: `pip install flash-attn --no-build-isolation` (requires CUDA toolkit matching your PyTorch build).

### Speculative Decoding (Stage 3)

For the LLM post-processing stage, speculative decoding can achieve 2-3x faster token generation by using a smaller draft model to predict tokens that the main model then verifies in parallel.

- **llama.cpp**: Supports speculative decoding natively via `--draft` flag or `draft_model` parameter
- **Draft model candidate**: Qwen3.5-1.7B Q4_K_M (~1GB VRAM) as draft for Qwen3.5-9B
- **Trade-off**: Requires fitting both models in VRAM simultaneously (~5.5GB + ~1GB = ~6.5GB, still within 8GB budget)
- **Implementation**: Add optional `speculative_decoding` config to Stage 3; evaluate whether the VRAM trade-off is worth the speed gain during profiling

---

## Distribution

### Option A: Docker (Recommended)

Single `docker run` command with all dependencies bundled, including CUDA runtime:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Install Python, pip, and project dependencies
COPY . /app
RUN pip install -e /app[vllm]
ENTRYPOINT ["python", "scripts/transcribe.py"]
```

Usage:
```bash
docker run --gpus all -v /path/to/audio:/data pipeline:latest /data/meeting.wav
```

Advantages: reproducible environment, no Python version conflicts, easy to share.

### Option B: uv (Fast Python Package Manager)

Use [uv](https://github.com/astral-sh/uv) for fast, deterministic installs with lockfile:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run directly (uv resolves dependencies automatically)
uv run scripts/transcribe.py meeting.wav

# Or create a locked environment
uv sync
uv run scripts/transcribe.py meeting.wav
```

Add `uv.lock` to the repository for reproducible installs. uv resolves and installs dependencies 10-100x faster than pip.

### Option C: PyInstaller / Nuitka (Single Binary)

For distributing to users without Python:

```bash
# PyInstaller
pyinstaller --onefile scripts/transcribe.py

# Nuitka (better optimization)
nuitka --standalone --onefile scripts/transcribe.py
```

Note: PyTorch + CUDA bundling produces large binaries (~2-5GB). Docker is generally preferred.

---

## Dependencies

### File: `pyproject.toml`

```toml
[project]
name = "transcription-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Audio processing
    "torchaudio>=2.0",              # https://pytorch.org/audio/
    "torch>=2.0",                   # https://pytorch.org/

    # VAD
    "silero-vad>=5.0",              # https://github.com/snakers4/silero-vad

    # Speaker diarization
    "pyannote.audio>=3.1",          # https://github.com/pyannote/pyannote-audio

    # ASR (vLLM backend is primary — see Language Evaluation section)
    "qwen-asr[vllm]",              # https://pypi.org/project/qwen-asr/
    "vllm>=0.14",                   # https://github.com/vllm-project/vllm

    # LLM inference
    "llama-cpp-python>=0.3",        # https://github.com/abetlen/llama-cpp-python

    # Utilities
    "pyyaml",
    "numpy",
    "scipy",
    "huggingface-hub",
]

[project.optional-dependencies]
# Minimal install without vLLM (uses transformers backend for ASR)
minimal = [
    "qwen-asr",                     # Without vLLM extra
]
flash-attn = [
    "flash-attn>=2.0",              # Flash Attention 2 for faster inference
]
```

---

## CLI Entry Point

### File: `scripts/transcribe.py`

```
Usage:
  python scripts/transcribe.py <audio_file> [--config config.yaml] [--output-dir output/]
  python scripts/transcribe.py meeting.mp4 --num-speakers 3
  python scripts/transcribe.py interview.wav --no-summary
```

### File: `scripts/enroll_speaker.py`

```
Usage:
  python scripts/enroll_speaker.py --name "Alice" --audio alice_sample1.wav alice_sample2.wav
  # Extracts and stores voice print embedding to speaker_profiles/Alice.npy
```

---

## Implementation Order

### Phase 1: Core Infrastructure
1. `src/gpu_utils.py` — GPU memory management
2. `src/audio_preprocessing.py` — audio loading and conversion
3. `config.yaml` — configuration file
4. `src/pipeline.py` — skeleton orchestrator with stage placeholders

### Phase 2: Diarization
5. `src/vad.py` — Silero VAD integration
6. `src/diarization.py` — pyannote diarization
7. `src/speaker_registry.py` — speaker enrollment and matching

### Phase 3: Transcription
8. `src/transcription.py` — Qwen3-ASR integration

### Phase 4: LLM Post-Processing
9. `src/llm_postprocess.py` — Qwen3.5 integration with DiarizationLM-inspired prompting

### Phase 5: Output and Polish
10. `src/output_formatter.py` — output formatting
11. `scripts/transcribe.py` — CLI entry point
12. `scripts/enroll_speaker.py` — speaker enrollment CLI
13. `tests/` — unit tests

---

## Verification Plan

### Test 1: Audio Preprocessing
- Load a multi-format audio file (MP3, MP4, WAV)
- Verify output is 16kHz mono torch.Tensor
- Verify normalization to [-1, 1]

### Test 2: VAD
- Process audio with known silence gaps
- Verify silence segments are removed
- Verify timestamp mapping is correct (map back to original positions)

### Test 3: Diarization
- Use a 2-speaker audio sample
- Verify pyannote returns 2 speaker labels
- Verify segment timestamps are reasonable

### Test 4: Speaker Registration
- Enroll a speaker with `enroll_speaker.py`
- Verify `.npy` file created in `speaker_profiles/`
- Run diarization on audio containing that speaker
- Verify speaker is correctly identified by name

### Test 5: ASR
- Transcribe a Chinese-English mixed audio segment
- Verify both languages are correctly recognized
- Compare output against known ground truth if available

### Test 6: LLM Post-Processing
- Feed a transcript with intentional speaker label errors
- Verify LLM corrects labels without modifying words
- Verify TPST safety check catches word modifications

### Test 7: End-to-End
- Run full pipeline on a real meeting recording (5-10 minutes)
- Verify all output formats (JSON, SRT, TXT) are generated
- Verify speaker names appear where enrolled
- Verify VRAM stays under 8GB at each stage (check logs)
- Measure total processing time

### Test 8: Crash Recovery
- Kill pipeline mid-Stage-2
- Restart — verify it picks up from Stage 2 using saved intermediate results

---

## References

### Models
- **Silero VAD v5**: [GitHub](https://github.com/snakers4/silero-vad) | [PyTorch Hub](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- **pyannote speaker-diarization-3.1**: [HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1) | [GitHub](https://github.com/pyannote/pyannote-audio) | [Benchmark](https://www.pyannote.ai/benchmark)
- **WeSpeaker ResNet34-LM**: [HuggingFace](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | [GitHub](https://github.com/wenet-e2e/wespeaker) | [Paper](https://www.researchgate.net/publication/371288447_Wespeaker_A_Research_and_Production_Oriented_Speaker_Embedding_Learning_Toolkit)
- **Qwen3-ASR-1.7B**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | [GitHub](https://github.com/QwenLM/Qwen3-ASR) | [Technical Report](https://arxiv.org/html/2601.21337v1) | [PyPI](https://pypi.org/project/qwen-asr/)
- **Qwen3.5-9B**: [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B) | [GitHub](https://github.com/QwenLM/Qwen3.5) | [GGUF (Unsloth)](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | [GGUF (Bartowski)](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF)

### Libraries
- **torchaudio**: [Docs](https://pytorch.org/audio/stable/index.html) | [torchaudio.load](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.load.html) | [Transforms](https://docs.pytorch.org/audio/stable/transforms.html)
- **llama-cpp-python**: [GitHub](https://github.com/abetlen/llama-cpp-python)
- **vLLM**: [GitHub](https://github.com/vllm-project/vllm) | [Qwen3-ASR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html) | [Qwen3.5 Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- **pyannote-core**: [Annotation API](https://pyannote.github.io/pyannote-core/structure.html#annotation)

### Research Papers
- **DiarizationLM**: [Paper (INTERSPEECH 2024)](https://arxiv.org/abs/2401.03506) | [GitHub](https://github.com/google/speaker-id/tree/master/DiarizationLM) | [HuggingFace Models](https://huggingface.co/google/DiarizationLM-13b-Fisher-v1)
- **LLM-Diarize-ASR-Agnostic**: [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167639325000391) | [GitHub](https://github.com/GeorgeEfstathiadis/LLM-Diarize-ASR-Agnostic)
- **Qwen3-ASR Technical Report**: [arXiv](https://arxiv.org/abs/2601.21337)
- **SpeakerLM (End-to-End)**: [arXiv](https://arxiv.org/abs/2508.06372)
- **JEDIS-LLM (Train Short, Infer Long)**: [arXiv](https://arxiv.org/abs/2511.16046)

### Reference Implementations
- **WhisperX** (ASR + diarization reference): [GitHub](https://github.com/m-bain/whisperX)
- **Transcription Stream** (full self-hosted solution): [GitHub](https://github.com/transcriptionstream/transcriptionstream)
- **qwen3-asr-swift** (Qwen3-ASR + pyannote on Apple Silicon): [GitHub](https://github.com/ivan-digital/qwen3-asr-swift)

### Formats
- **RTTM Specification**: [LDC Documentation](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
- **SRT Format**: [Wikipedia](https://en.wikipedia.org/wiki/SubRip#SubRip_file_format)

### Datasets (for testing/benchmarking)
- **VoxCeleb2**: [Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- **Fisher English**: [LDC](https://catalog.ldc.upenn.edu/LDC2004T19)
- **AliMeeting**: [OpenSLR](https://www.openslr.org/119/)
- **AISHELL-1**: [OpenSLR](https://www.openslr.org/33/)
