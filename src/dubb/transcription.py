"""Transcription utilities using Faster Whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from faster_whisper import BatchedInferencePipeline, WhisperModel

from dubb.schemas import Segment, Word

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
    batch_size: int = 8,
    word_timestamps: bool = True,
) -> TranscriptionResult:
    """Transcribe source audio and return timestamped segments plus detected language.

    Uses faster-whisper's batched inference pipeline when ``batch_size`` is
    greater than one, which can substantially reduce transcription time on
    both GPU and CPU with no accuracy trade-off. Word-level timestamps are
    requested by default so downstream stages can perform finer-grained
    alignment than segment-level timing alone allows.
    """
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

    transcribe_kwargs = {"vad_filter": True, "beam_size": 5, "word_timestamps": word_timestamps}
    if batch_size > 1:
        batched_model = BatchedInferencePipeline(model=model)
        segments, info = batched_model.transcribe(str(audio_path), batch_size=batch_size, **transcribe_kwargs)
    else:
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    parsed_segments = [
        Segment(
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text.strip(),
            words=_parse_words(segment.words),
        )
        for segment in segments
        if segment.text.strip()
    ]
    return TranscriptionResult(segments=parsed_segments, source_language=info.language or "auto")


def _parse_words(words: list[object] | None) -> list[Word] | None:
    """Convert faster-whisper word objects into serializable `Word` models."""
    if not words:
        return None
    return [
        Word(
            start=float(word.start),  # type: ignore[attr-defined]
            end=float(word.end),  # type: ignore[attr-defined]
            word=word.word.strip(),  # type: ignore[attr-defined]
            probability=float(getattr(word, "probability", 1.0)),
        )
        for word in words
    ]


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