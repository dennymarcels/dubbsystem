"""Transcription utilities using Faster Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from faster_whisper import WhisperModel

from dubb.schemas import Segment


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
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
    parsed_segments = [
        Segment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    return TranscriptionResult(segments=parsed_segments, source_language=info.language or "auto")