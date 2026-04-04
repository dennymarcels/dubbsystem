"""Transcription utilities using Faster Whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from faster_whisper import WhisperModel

from dubb.schemas import Segment

logger = logging.getLogger(__name__)


class TranscriptionResult(NamedTuple):
    """Container for transcript segments and detected source language."""

    segments: list[Segment]
    source_language: str


def transcribe_with_timestamps(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
) -> TranscriptionResult:
    """Transcribe source audio and return timestamped segments plus detected language."""
    runtime_device = device
    runtime_compute_type = compute_type
    try:
        model = WhisperModel(model_name, device=runtime_device, compute_type=runtime_compute_type)
    except RuntimeError as exc:
        if device != "cuda" or not _is_cuda_runtime_failure(exc):
            raise
        runtime_device = "cpu"
        runtime_compute_type = "int8"
        logger.warning(
            "Faster Whisper CUDA initialization failed (%s). Falling back to CPU transcription with compute_type=%s.",
            exc,
            runtime_compute_type,
        )
        model = WhisperModel(model_name, device=runtime_device, compute_type=runtime_compute_type)

    segments, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
    parsed_segments = [
        Segment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    return TranscriptionResult(segments=parsed_segments, source_language=info.language or "auto")


def _is_cuda_runtime_failure(error: RuntimeError) -> bool:
    """Return whether the runtime error indicates a CUDA backend compatibility problem."""
    message = str(error).lower()
    indicators = [
        "cuda failed",
        "cuda driver version is insufficient",
        "failed to create cublas handle",
        "cudnn",
        "ctranslate2",
    ]
    return any(indicator in message for indicator in indicators)