# Security Policy

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email security concerns to the maintainer via GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/yuhanwang14/ASR-Pipeline/security) of this repository
2. Click "Report a vulnerability"
3. Provide a description, steps to reproduce, and impact assessment

### What to expect

- Acknowledgment within 72 hours
- A fix will be developed privately before public disclosure
- Credit in the security advisory (unless you prefer to remain anonymous)

## Scope

Security-relevant areas of this project include:

- **HuggingFace token handling** -- tokens must only be read from environment variables (`HF_TOKEN` or `ASR_HF_TOKEN`), never hardcoded or logged
- **Subprocess calls** -- `ffmpeg` is invoked via `subprocess.run()` with fixed arguments; user input must not be interpolated into commands
- **File path handling** -- audio file paths and output directories come from user input and must be handled safely
- **Model downloads** -- models are downloaded from HuggingFace Hub; verify you trust the model source

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Dependencies

This project depends on PyTorch, pyannote.audio, vLLM, llama-cpp-python, and other packages. We recommend keeping dependencies up to date. If you discover a vulnerability in a dependency that affects this project, please report it.
