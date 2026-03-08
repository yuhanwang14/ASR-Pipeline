#!/usr/bin/env python3
"""CLI entry point for the transcription pipeline."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (auto-detect if not set)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed stage",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Output formats (json srt rttm txt)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Import here to avoid slow imports on --help
    from src.config import load_config
    from src.logging_config import setup_logging
    from src.output_formatter import save_outputs
    from src.pipeline import run_pipeline

    setup_logging(verbose=args.verbose)

    try:
        config = load_config(args.config)

        # Apply CLI overrides
        if args.num_speakers is not None:
            config["diarization"]["num_speakers"] = args.num_speakers
        if args.formats:
            config["output"]["formats"] = args.formats
        if args.output_dir:
            config["output"]["output_dir"] = args.output_dir

        output_dir = config["output"]["output_dir"]

        # Clear cached stage files unless --resume is set
        if not args.resume:
            from pathlib import Path

            out_path = Path(output_dir)
            if out_path.exists():
                for stage_file in out_path.glob("stage_*.json"):
                    stage_file.unlink()

        result = run_pipeline(args.audio_file, config=config, output_dir=output_dir)
        saved = save_outputs(result, config, args.audio_file)

        print("\nTranscription complete!")
        print("Output files:")
        for path in saved:
            print(f"  {path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        if "VRAM" in str(e) or "OutOfMemory" in str(e):
            print(f"GPU memory error: {e}", file=sys.stderr)
            print(
                "Try reducing batch size or closing other GPU applications.",
                file=sys.stderr,
            )
        else:
            print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
