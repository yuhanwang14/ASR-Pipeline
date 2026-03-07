"""Speaker voice print enrollment and matching."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("asr_pipeline")


def _get_profiles_dir(config: dict) -> Path:
    """Get speaker profiles directory from config, creating it if needed."""
    diar_cfg = config.get("diarization", {})
    profiles_dir = Path(diar_cfg.get("speaker_profiles_dir", "speaker_profiles/"))
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def enroll_speaker(name: str, audio_paths: list[str], config: dict) -> None:
    """Enroll a speaker by extracting embeddings from audio files and saving averaged embedding.

    Lazy-imports pyannote.audio Model and Inference. Loads the WeSpeaker model
    specified in config, extracts one embedding per audio file, averages them,
    and saves the result to speaker_profiles_dir/<name>.npy.

    Args:
        name: Speaker name (used as filename stem).
        audio_paths: List of audio file paths for this speaker.
        config: Pipeline configuration dictionary.

    Raises:
        ValueError: If audio_paths is empty.
        FileNotFoundError: If any audio file does not exist.
    """
    if not audio_paths:
        raise ValueError("At least one audio file is required for enrollment.")

    for p in audio_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

    import torch
    from pyannote.audio import Inference, Model

    diar_cfg = config.get("diarization", {})
    embedding_model_name = diar_cfg.get(
        "embedding_model", "pyannote/wespeaker-voxceleb-resnet34-LM"
    )
    hf_token = diar_cfg.get("hf_token")

    logger.info("Loading embedding model: %s", embedding_model_name)
    model = Model.from_pretrained(embedding_model_name, use_auth_token=hf_token)
    inference = Inference(model, window="whole")
    if torch.cuda.is_available():
        inference.to(torch.device("cuda"))

    embeddings = []
    for audio_path in audio_paths:
        logger.info("Extracting embedding from: %s", audio_path)
        embedding = inference(audio_path)
        embeddings.append(embedding)

    avg_embedding = np.mean(embeddings, axis=0)

    profiles_dir = _get_profiles_dir(config)
    save_path = profiles_dir / f"{name}.npy"
    np.save(save_path, avg_embedding)
    logger.info("Saved speaker profile for '%s' to %s", name, save_path)

    # Clean up
    from src.gpu_utils import unload_model

    unload_model(model)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def match_speakers(segments: list[dict], config: dict) -> list[dict]:
    """Match anonymous speaker labels to enrolled profiles using cosine similarity.

    For each unique speaker in segments, looks for an "embedding" field to
    compute an average embedding, then compares against all saved profiles.
    If the best match exceeds the configured threshold, replaces the speaker
    label with the enrolled name.

    Args:
        segments: List of diarization segments, each with at least
            {"start", "end", "speaker"}. May optionally include "embedding".
        config: Pipeline configuration dictionary.

    Returns:
        Updated segments with speaker labels replaced where a match is found.
    """
    diar_cfg = config.get("diarization", {})
    threshold = diar_cfg.get("match_threshold", 0.75)
    profiles_dir = _get_profiles_dir(config)

    # Load all enrolled profiles
    profiles: dict[str, np.ndarray] = {}
    for profile_path in profiles_dir.glob("*.npy"):
        speaker_name = profile_path.stem
        profiles[speaker_name] = np.load(profile_path)

    if not profiles:
        logger.info("No enrolled speaker profiles found; skipping matching.")
        return segments

    # Group segments by anonymous speaker label and collect embeddings
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        speaker = seg["speaker"]
        if "embedding" in seg and seg["embedding"] is not None:
            speaker_embeddings.setdefault(speaker, []).append(np.asarray(seg["embedding"]))

    # Build label mapping: anonymous label -> real name
    label_map: dict[str, str] = {}
    for speaker_label, embs in speaker_embeddings.items():
        avg_emb = np.mean(embs, axis=0)

        best_name = None
        best_sim = -1.0
        for name, profile_emb in profiles.items():
            sim = _cosine_similarity(avg_emb, profile_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold and best_name is not None:
            label_map[speaker_label] = best_name
            logger.info(
                "Matched '%s' -> '%s' (similarity: %.3f)",
                speaker_label,
                best_name,
                best_sim,
            )
        else:
            logger.info(
                "No match for '%s' (best: %.3f < threshold %.2f)",
                speaker_label,
                best_sim,
                threshold,
            )

    # Apply label mapping
    updated = []
    for seg in segments:
        new_seg = dict(seg)
        if new_seg["speaker"] in label_map:
            new_seg["speaker"] = label_map[new_seg["speaker"]]
        updated.append(new_seg)

    return updated


def list_enrolled_speakers(config: dict) -> list[str]:
    """List all enrolled speaker names.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Sorted list of enrolled speaker names.
    """
    profiles_dir = _get_profiles_dir(config)
    return sorted(p.stem for p in profiles_dir.glob("*.npy"))


def delete_speaker(name: str, config: dict) -> None:
    """Delete an enrolled speaker profile.

    Args:
        name: Speaker name to delete.
        config: Pipeline configuration dictionary.

    Raises:
        FileNotFoundError: If the speaker profile does not exist.
    """
    profiles_dir = _get_profiles_dir(config)
    profile_path = profiles_dir / f"{name}.npy"
    if not profile_path.exists():
        raise FileNotFoundError(f"Speaker profile not found: {profile_path}")
    profile_path.unlink()
    logger.info("Deleted speaker profile: %s", name)
