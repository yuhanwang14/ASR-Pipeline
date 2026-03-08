## What does this PR do?

Brief description of the changes.

## Why?

Motivation or issue reference (e.g., Fixes #123).

## How to test

Steps to verify the change works correctly.

## Checklist

- [ ] Tests pass (`uv run pytest tests/ -k "not integration and not gpu"`)
- [ ] Linting passes (`uv run ruff check src/ tests/ scripts/`)
- [ ] No new GPU dependencies at module level (lazy imports only)
- [ ] Models are unloaded after use (VRAM cleanup)
- [ ] Documentation updated (if applicable)
