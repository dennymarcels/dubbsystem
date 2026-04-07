"""Tests for media helper behavior."""

from __future__ import annotations

from pathlib import Path

import ffmpeg
from pydub import AudioSegment
from pydub.generators import Sine

from dubb.media import condense_speech_pauses, create_voice_sample, normalize_audio_timing
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


def test_create_voice_sample_filters_to_dominant_speaker_cluster(tmp_path: Path) -> None:
    """Ensure acoustically inconsistent segments are filtered out from the automatic speaker reference."""
    source_audio = tmp_path / "source.wav"
    output_sample = tmp_path / "speaker_sample.wav"

    audio = (
        Sine(220).to_audio_segment(duration=2_000).apply_gain(-10)
        + AudioSegment.silent(duration=400)
        + Sine(220).to_audio_segment(duration=2_000).apply_gain(-11)
        + AudioSegment.silent(duration=400)
        + Sine(880).to_audio_segment(duration=2_000).apply_gain(-8)
    )
    audio.export(source_audio, format="wav")

    segments = [
        Segment(start=0.0, end=2.0, text="Primary speaker one with stable tone."),
        Segment(start=2.4, end=4.4, text="Primary speaker two with similar voice."),
        Segment(start=4.8, end=6.8, text="Distractor speaker with very different voice."),
    ]

    result = create_voice_sample(
        source_audio=source_audio,
        output_sample=output_sample,
        duration_seconds=4,
        segments=segments,
    )

    assert result.output_path.exists()
    assert len(result.selection) >= 2
    assert all(item["start_ms"] < 4_800 for item in result.selection)
    assert all("speaker_similarity" in item for item in result.selection)


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


def test_normalize_audio_timing_can_hard_fit_when_overflow_is_not_allowed(tmp_path: Path) -> None:
    """Ensure timing normalization can force a clip into the available slot when overlap must be avoided."""
    source_audio = tmp_path / "long.wav"
    output_audio = tmp_path / "fitted.wav"

    Sine(440).to_audio_segment(duration=2_000).export(source_audio, format="wav")

    normalize_audio_timing(
        source_audio=source_audio,
        output_audio=output_audio,
        target_duration=1.0,
        min_tempo_factor=0.9,
        max_tempo_factor=2.5,
        allow_overflow=False,
    )

    output_duration = float(ffmpeg.probe(str(output_audio))["format"]["duration"])
    assert output_duration <= 1.05


def test_condense_speech_pauses_shortens_long_pauses(tmp_path: Path) -> None:
    """Ensure pause condensation reduces avoidable silence before hard fitting a chunk."""
    source_audio = tmp_path / "with_pauses.wav"
    output_audio = tmp_path / "condensed.wav"

    audio = Sine(440).to_audio_segment(duration=500) + AudioSegment.silent(duration=1_500) + Sine(440).to_audio_segment(duration=500)
    audio.export(source_audio, format="wav")

    condense_speech_pauses(source_audio, output_audio)

    original_duration = float(ffmpeg.probe(str(source_audio))["format"]["duration"])
    condensed_duration = float(ffmpeg.probe(str(output_audio))["format"]["duration"])
    assert condensed_duration < original_duration