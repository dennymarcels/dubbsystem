"""Tests for media helper behavior."""

from __future__ import annotations

from pathlib import Path

import ffmpeg
from pydub import AudioSegment
from pydub.generators import Sine

from dubb.media import create_voice_sample, normalize_audio_timing
from dubb.schemas import Segment


def test_create_voice_sample_prefers_ranked_segments(tmp_path: Path) -> None:
    """Ensure the speaker sample is built from ranked transcript segments instead of a naive first window."""
    source_audio = tmp_path / "source.wav"
    output_sample = tmp_path / "speaker_sample.wav"

    audio = (
        AudioSegment.silent(duration=2_000)
        + Sine(220).to_audio_segment(duration=3_000).apply_gain(-10)
        + AudioSegment.silent(duration=1_500)
        + Sine(330).to_audio_segment(duration=3_000).apply_gain(-12)
    )
    audio.export(source_audio, format="wav")

    segments = [
        Segment(start=2.0, end=5.0, text="This is a strong reference segment."),
        Segment(start=6.5, end=9.5, text="This is another clear reference segment."),
    ]

    result = create_voice_sample(
        source_audio=source_audio,
        output_sample=output_sample,
        duration_seconds=4,
        segments=segments,
    )

    assert result.output_path.exists()
    assert result.selection
    assert all(item["strategy"] == "transcript-ranked" for item in result.selection)
    assert all(item["start_ms"] >= 2_000 for item in result.selection)


def test_normalize_audio_timing_preserves_full_speech_when_clip_is_longer(tmp_path: Path) -> None:
    """Ensure timing normalization does not cut off synthesized speech when it exceeds the target window."""
    source_audio = tmp_path / "long.wav"
    output_audio = tmp_path / "aligned.wav"

    Sine(440).to_audio_segment(duration=2_000).export(source_audio, format="wav")

    normalize_audio_timing(
        source_audio=source_audio,
        output_audio=output_audio,
        target_duration=1.0,
        min_tempo_factor=0.9,
        max_tempo_factor=1.15,
    )

    output_duration = float(ffmpeg.probe(str(output_audio))["format"]["duration"])
    assert output_duration > 1.0