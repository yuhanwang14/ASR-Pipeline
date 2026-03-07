"""Structured logging setup for the ASR pipeline."""

import logging
import sys
from pathlib import Path


class _ColorFormatter(logging.Formatter):
    """Colorized log formatter for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


# Map logger name suffixes to stage labels
_STAGE_LABELS = {
    "vad": "VAD",
    "diarization": "DIARIZATION",
    "transcription": "ASR",
    "llm": "LLM",
    "output": "OUTPUT",
    "pipeline": "PIPELINE",
}


class _StageFormatter(logging.Formatter):
    """Formatter that includes a [STAGE] tag based on the logger name."""

    def __init__(self, fmt: str, datefmt: str | None = None, colorize: bool = False):
        super().__init__(fmt, datefmt)
        self._colorize = colorize
        self._color_formatter = _ColorFormatter(fmt, datefmt) if colorize else None

    def format(self, record: logging.LogRecord) -> str:
        # Determine stage label from logger name
        name = record.name
        stage = "GENERAL"
        for suffix, label in _STAGE_LABELS.items():
            if name.endswith(f".{suffix}") or name == f"asr_pipeline.{suffix}":
                stage = label
                break
        if name == "asr_pipeline":
            stage = "PIPELINE"

        record.stage = stage

        if self._color_formatter:
            return self._color_formatter.format(record)
        return super().format(record)


def setup_logging(verbose: bool = False) -> None:
    """Set up structured logging for the pipeline.

    Format: [TIMESTAMP] [STAGE] [LEVEL] message

    - Console handler with colorized output (if terminal supports it)
    - File handler to output/pipeline.log
    - Named loggers per stage: asr_pipeline.vad, asr_pipeline.diarization, etc.
    - Default level: INFO, verbose: DEBUG

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    fmt = "[%(asctime)s] [%(stage)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root pipeline logger
    root_logger = logging.getLogger("asr_pipeline")
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console_formatter = _StageFormatter(fmt, datefmt, colorize=use_color)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    log_dir = Path("output")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = _StageFormatter(fmt, datefmt, colorize=False)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False
