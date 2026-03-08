# Contributing to ASR Pipeline

Thank you for your interest in contributing! This document explains how to get started.

## Getting Started

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An NVIDIA GPU is **not** required for development -- all unit tests run on CPU with mock backends

### Development Setup

```bash
git clone https://github.com/yuhanwang14/ASR-Pipeline.git
cd ASR-Pipeline
uv sync --group dev
```

Verify your setup:

```bash
uv run pytest tests/ -k "not integration and not gpu"
uv run ruff check src/ tests/ scripts/
```

Both commands should pass with zero errors.

## How to Contribute

### Reporting Bugs

Open an [issue](https://github.com/yuhanwang14/ASR-Pipeline/issues) using the bug report template. Include:

- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU model, VRAM)
- Relevant log output (run with `--verbose`)

### Suggesting Features

Open an issue to discuss the idea before writing code. This saves everyone time if the feature is out of scope or if there's a preferred approach.

### Submitting Pull Requests

1. **Open an issue first** to discuss the change (unless it's a trivial fix)
2. Fork the repo and create a branch from `master`
3. Make your changes
4. Add or update tests -- aim for the same test patterns used in `tests/`
5. Run the full check:
   ```bash
   uv run ruff check src/ tests/ scripts/
   uv run ruff format --check src/ tests/ scripts/
   uv run pytest tests/ -k "not integration and not gpu"
   ```
6. Open a PR against `master`

### What We Look For in PRs

- **Tests pass** -- CI runs ruff lint/format checks and the full CPU test suite
- **Tests included** -- new features need tests; bug fixes need a regression test
- **Focused scope** -- one logical change per PR
- **No unrelated changes** -- don't reformat code you didn't modify
- **Backwards compatible** -- breaking changes need discussion first

## Code Style

- **Formatter**: [ruff](https://docs.astral.sh/ruff/) with 100-char line length
- **Linter**: ruff with pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear rules
- **Imports**: lazy imports for heavy dependencies (`torch`, `torchaudio`, `pyannote.audio`, etc.) -- import inside functions, not at module level
- **Type hints**: use them for public functions; avoid top-level imports of `torch` types (use `TYPE_CHECKING` guard)
- **Protocols**: GPU backends use `typing.Protocol` for dependency injection -- maintain this pattern

Run `uv run ruff format src/ tests/ scripts/` before committing to auto-fix style issues.

## Architecture Notes

If you're contributing to a specific stage, keep these design constraints in mind:

- **VRAM budget**: each stage must stay under 8 GB peak. Always clean up GPU memory after use.
- **Serial execution**: stages run one at a time. Don't assume other models are loaded.
- **Crash recovery**: if your change adds a new stage or modifies stage output, ensure intermediate results serialize to JSON via `intermediate.py`.
- **LLM safety**: Stage 3 must never alter transcribed words. Use JSON error-pair output, not full-transcript rewriting.

## Test Organization

- `tests/test_*.py` -- one test file per source module
- GPU tests are marked with `@pytest.mark.gpu` and excluded from CI
- Integration tests are marked with `@pytest.mark.integration`
- All GPU backends have mock implementations for CPU-only testing

## Questions?

Open an issue or start a [discussion](https://github.com/yuhanwang14/ASR-Pipeline/discussions) on GitHub.
