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
    if target_duration <= 0:
        raise ValueError("target_duration must be greater than zero")

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    current_duration = float(ffmpeg.probe(str(source_audio))["format"]["duration"])
    ratio = current_duration / target_duration
    ratio = min(max(ratio, 0.5), 100.0)

    factors: list[float] = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)

    stream = ffmpeg.input(str(source_audio)).audio
    for factor in factors:
        stream = stream.filter("atempo", factor)
    (
        stream.output(str(output_audio), ac=1, ar=24_000)
        .overwrite_output()
        .run(quiet=True)
    )
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