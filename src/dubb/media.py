"""Media processing helpers built on FFmpeg and pydub."""

from __future__ import annotations

from pathlib import Path

import ffmpeg
from pydub import AudioSegment


def extract_audio(input_video: Path, output_audio: Path, sample_rate: int) -> Path:
    """Extract a mono WAV track from a video file."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.input(str(input_video))
        .output(str(output_audio), ac=1, ar=sample_rate, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_audio


def create_voice_sample(source_audio: Path, output_sample: Path, duration_seconds: int) -> Path:
    """Create a speaker reference sample from the source audio."""
    output_sample.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(source_audio)
    clip = audio[: duration_seconds * 1000]
    clip.export(output_sample, format="wav")
    return output_sample


def fit_audio_to_duration(source_audio: Path, output_audio: Path, target_duration: float) -> Path:
    """Time-scale an audio clip to match a target duration as closely as FFmpeg allows."""
    return normalize_audio_timing(
        source_audio=source_audio,
        output_audio=output_audio,
        target_duration=target_duration,
        min_tempo_factor=0.5,
        max_tempo_factor=100.0,
    )


def normalize_audio_timing(
    source_audio: Path,
    output_audio: Path,
    target_duration: float,
    min_tempo_factor: float,
    max_tempo_factor: float,
) -> Path:
    """Adjust clip pacing within bounded tempo limits, then trim or pad to the target duration."""
    if target_duration <= 0:
        raise ValueError("target_duration must be greater than zero")
    if min_tempo_factor <= 0 or max_tempo_factor <= 0:
        raise ValueError("tempo factors must be greater than zero")
    if min_tempo_factor > max_tempo_factor:
        raise ValueError("min_tempo_factor cannot be greater than max_tempo_factor")

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    current_duration = float(ffmpeg.probe(str(source_audio))["format"]["duration"])
    ratio = current_duration / target_duration
    ratio = min(max(ratio, min_tempo_factor), max_tempo_factor)

    factors: list[float] = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)

    tempo_adjusted_audio = output_audio.with_name(f"{output_audio.stem}_tempo{output_audio.suffix}")

    stream = ffmpeg.input(str(source_audio)).audio
    for factor in factors:
        stream = stream.filter("atempo", factor)
    (
        stream.output(str(tempo_adjusted_audio), ac=1, ar=24_000)
        .overwrite_output()
        .run(quiet=True)
    )

    clip = AudioSegment.from_file(tempo_adjusted_audio)
    target_duration_ms = int(target_duration * 1000)
    if len(clip) > target_duration_ms:
        clip = clip[:target_duration_ms]
    elif len(clip) < target_duration_ms:
        clip += AudioSegment.silent(duration=target_duration_ms - len(clip))
    clip.export(output_audio, format="wav")
    tempo_adjusted_audio.unlink(missing_ok=True)
    return output_audio


def overlay_segments(segments: list[tuple[Path, float]], output_audio: Path, duration_seconds: float) -> Path:
    """Overlay timestamped segment files into a single audio track."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    canvas = AudioSegment.silent(duration=int(duration_seconds * 1000))
    for segment_path, start_seconds in segments:
        clip = AudioSegment.from_file(segment_path)
        canvas = canvas.overlay(clip, position=int(start_seconds * 1000))
    canvas.export(output_audio, format="wav")
    return output_audio


def mux_audio_with_video(input_video: Path, dubbed_audio: Path, output_video: Path) -> Path:
    """Replace the video audio track with the dubbed track and export MP4."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    video_input = ffmpeg.input(str(input_video))
    audio_input = ffmpeg.input(str(dubbed_audio))
    (
        ffmpeg.output(
            video_input.video,
            audio_input.audio,
            str(output_video),
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_video