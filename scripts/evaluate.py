#!/usr/bin/env python3
"""Evaluate ASR pipeline output against a reference transcript.

Usage:
    # Evaluate cached output (fast):
    uv run scripts/evaluate.py --output "output/zoom-mar-7/Zoom - Mar 7.json" \
        --reference "tests/fixtures/Zoom - Mar 7.md"

    # Run pipeline + evaluate:
    uv run scripts/evaluate.py "tests/fixtures/Zoom - Mar 7.m4a" \
        --reference "tests/fixtures/Zoom - Mar 7.md"
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR pipeline output")
    parser.add_argument("audio_file", nargs="?", help="Audio file (runs pipeline first)")
    parser.add_argument("--output", help="Pre-computed pipeline JSON output (skip pipeline)")
    parser.add_argument("--reference", required=True, help="Reference transcript (.md)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config file")
    parser.add_argument("--output-dir", default=None, help="Pipeline output directory")

    args = parser.parse_args()

    if not args.audio_file and not args.output:
        parser.error("Provide either an audio file or --output path")

    from src.eval_metrics import (
        compute_cer,
        compute_cpwer,
        compute_wer,
        load_pipeline_output,
        normalize_for_eval,
        parse_reference_transcript,
    )

    # Load reference
    ref_text = Path(args.reference).read_text(encoding="utf-8")
    ref_segments = parse_reference_transcript(ref_text)
    if not ref_segments:
        print("Error: no segments found in reference transcript", file=sys.stderr)
        sys.exit(1)

    # Load or generate hypothesis
    if args.output:
        hyp_segments = load_pipeline_output(args.output)
        src_label = args.output
    else:
        from src.config import load_config
        from src.logging_config import setup_logging
        from src.pipeline import run_pipeline

        setup_logging(verbose=True)
        config = load_config(args.config)
        if args.output_dir:
            config["output"]["output_dir"] = args.output_dir
        result = run_pipeline(args.audio_file, config=config)
        hyp_segments = [
            s for s in result["segments"]
            if s.get("text", "").strip() and not s.get("text", "").startswith("<think>")
        ]
        src_label = args.audio_file

    if not hyp_segments:
        print("Error: no segments found in pipeline output", file=sys.stderr)
        sys.exit(1)

    # Text-level metrics (speaker-agnostic)
    ref_full = " ".join(normalize_for_eval(s["text"]) for s in ref_segments)
    hyp_full = " ".join(normalize_for_eval(s["text"]) for s in hyp_segments)
    cer = compute_cer(ref_full, hyp_full)
    wer = compute_wer(ref_full, hyp_full)

    # Speaker-aware metrics
    cpwer_result = compute_cpwer(ref_segments, hyp_segments)

    # Print scorecard
    ref_speakers = sorted(set(s["speaker"] for s in ref_segments))
    hyp_speakers = sorted(set(s["speaker"] for s in hyp_segments))

    print("=== ASR Pipeline Evaluation ===")
    print(f"Reference:  {args.reference} ({len(ref_speakers)} speakers, {len(ref_segments)} segments)")
    print(f"Hypothesis: {src_label} ({len(hyp_speakers)} speakers, {len(hyp_segments)} segments)")
    print()
    print("Text Quality (speaker-agnostic):")
    print(f"  CER: {cer:6.1%}")
    print(f"  WER: {wer:6.1%}")
    print()
    print("Speaker Diarization (cpWER):")
    mapping = cpwer_result["mapping"]
    mapping_str = ", ".join(f"{h}\u2192{r}" for h, r in sorted(mapping.items()))
    print(f"  Optimal mapping: {mapping_str}")
    print(f"  cpWER: {cpwer_result['cpwer']:6.1%}")
    print()
    if cpwer_result["per_speaker"]:
        print("  Per-speaker:")
        for ref_sp, metrics in sorted(cpwer_result["per_speaker"].items()):
            hyp_sp = metrics["hyp_speaker"]
            print(
                f"    {ref_sp} (\u2192{hyp_sp}):"
                f"  CER={metrics['cer']:.1%}"
                f"  WER={metrics['wer']:.1%}"
            )


if __name__ == "__main__":
    main()
