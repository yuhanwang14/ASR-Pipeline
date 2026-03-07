#!/usr/bin/env python3
"""CLI for managing speaker voice prints."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Manage speaker voice prints")
    parser.add_argument("--name", help="Speaker name to enroll or delete")
    parser.add_argument("--audio", nargs="+", help="Audio files for enrollment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    parser.add_argument("--delete", action="store_true", help="Delete speaker profile")

    args = parser.parse_args()

    from src.config import load_config

    config = load_config(args.config)

    from src.speaker_registry import (
        delete_speaker,
        enroll_speaker,
        list_enrolled_speakers,
    )

    if args.list:
        speakers = list_enrolled_speakers(config)
        if speakers:
            print("Enrolled speakers:")
            for s in speakers:
                print(f"  - {s}")
        else:
            print("No speakers enrolled.")
    elif args.delete and args.name:
        delete_speaker(args.name, config)
        print(f"Deleted speaker: {args.name}")
    elif args.name and args.audio:
        enroll_speaker(args.name, args.audio, config)
        print(f"Enrolled speaker: {args.name}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
