"""ASR evaluation metrics: CER, WER, cpWER against reference transcripts."""

import json
import re
from pathlib import Path

_SPEAKER_LINE_RE = re.compile(r"^\*\*Speaker (\d+):\*\*\s*(.+)$")
_TIMESTAMP_RE = re.compile(r"^\(\d+:\d+\)$")


def parse_reference_transcript(markdown: str) -> list[dict]:
    """Parse Gemini-style markdown transcript into segments.

    Expected format: ``**Speaker N:** text`` lines separated by blank lines.
    Skips blank lines, ``**Transcript**`` header, and ``(M:SS)`` timestamps.
    """
    segments: list[dict] = []
    for line in markdown.strip().splitlines():
        line = line.strip()
        if not line or _TIMESTAMP_RE.match(line):
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if m:
            speaker = f"Speaker {m.group(1)}"
            text = m.group(2).strip()
            if text:
                segments.append({"speaker": speaker, "text": text})
    return segments


def load_pipeline_output(path: str | Path) -> list[dict]:
    """Load pipeline JSON output and return filtered segments.

    Filters out empty-text segments and segments with leaked ``<think>`` blocks.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    segments = data.get("segments", [])
    return [
        s for s in segments
        if s.get("text", "").strip() and not s.get("text", "").startswith("<think>")
    ]
