"""Media processing helpers built on FFmpeg and pydub."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, NamedTuple

import ffmpeg
from pydub import AudioSegment, effects, silence

from dubb.schemas import Segment


class VoiceSampleResult(NamedTuple):
    """Result of building a cleaned speaker reference sample."""

    output_path: Path
    selection: list[dict[str, Any]]


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


def create_voice_sample(
    source_audio: Path,
    output_sample: Path,
    duration_seconds: int,
    segments: list[Segment] | None = None,
) -> VoiceSampleResult:
    """Create a cleaned speaker reference sample from the best transcript segments."""
    output_sample.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(source_audio)
    target_duration_ms = duration_seconds * 1000
    selection = _select_voice_sample_segments(audio, segments, target_duration_ms)

    cleaned_clips: list[AudioSegment] = []
    normalized_selection: list[dict[str, Any]] = []
    accumulated_duration_ms = 0
    for item in selection:
        if accumulated_duration_ms >= target_duration_ms:
            break

        remaining_ms = target_duration_ms - accumulated_duration_ms
        raw_clip = audio[item["start_ms"] : item["end_ms"]]
        cleaned_clip = _cleanup_voice_sample_clip(raw_clip)
        if len(cleaned_clip) <= 0:
            continue

        cleaned_clip = cleaned_clip[:remaining_ms]
        cleaned_clips.append(cleaned_clip)
        normalized_selection.append(
            {
                **item,
                "selected_duration_ms": len(cleaned_clip),
            }
        )
        accumulated_duration_ms += len(cleaned_clip)

    if not cleaned_clips:
        fallback_clip = _cleanup_voice_sample_clip(audio[:target_duration_ms])
        cleaned_clips.append(fallback_clip[:target_duration_ms])
        normalized_selection = [
            {
                "start_ms": 0,
                "end_ms": min(len(audio), target_duration_ms),
                "selected_duration_ms": len(cleaned_clips[0]),
                "score": 0.0,
                "text": "",
                "strategy": "fallback-first-window",
            }
        ]

    combined_clip = AudioSegment.empty()
    for cleaned_clip in cleaned_clips:
        combined_clip += cleaned_clip

    combined_clip = _cleanup_voice_sample_clip(combined_clip)[:target_duration_ms]
    combined_clip.export(output_sample, format="wav")
    return VoiceSampleResult(output_path=output_sample, selection=normalized_selection)


def _select_voice_sample_segments(
    audio: AudioSegment,
    segments: list[Segment] | None,
    target_duration_ms: int,
) -> list[dict[str, Any]]:
    """Score transcript segments and pick the best non-empty windows for the speaker sample."""
    if not segments:
        return [
            {
                "start_ms": 0,
                "end_ms": min(len(audio), target_duration_ms),
                "score": 0.0,
                "text": "",
                "strategy": "fallback-first-window",
            }
        ]

    candidates: list[dict[str, Any]] = []
    for segment in segments:
        start_ms = max(0, int(segment.start * 1000))
        end_ms = min(len(audio), int(segment.end * 1000))
        duration_ms = end_ms - start_ms
        if duration_ms < 1200:
            continue

        clip = audio[start_ms:end_ms]
        score = _score_voice_sample_clip(clip, segment.text)
        candidates.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "score": score,
                "text": segment.text,
                "strategy": "transcript-ranked",
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    accumulated_duration_ms = 0
    for candidate in candidates:
        if accumulated_duration_ms >= target_duration_ms:
            break
        segment_duration_ms = candidate["end_ms"] - candidate["start_ms"]
        if segment_duration_ms <= 0:
            continue
        selected.append(candidate)
        accumulated_duration_ms += segment_duration_ms

    if not selected:
        return [
            {
                "start_ms": 0,
                "end_ms": min(len(audio), target_duration_ms),
                "score": 0.0,
                "text": "",
                "strategy": "fallback-first-window",
            }
        ]
    return selected


def _score_voice_sample_clip(clip: AudioSegment, text: str) -> float:
    """Compute a heuristic score for how suitable a clip is as a voice reference."""
    clip_length_ms = max(len(clip), 1)
    silence_threshold = _voice_sample_silence_threshold(clip)
    speech_ranges = silence.detect_nonsilent(clip, min_silence_len=250, silence_thresh=silence_threshold)
    speech_duration_ms = sum(end - start for start, end in speech_ranges)
    speech_ratio = speech_duration_ms / clip_length_ms
    word_count = len(text.split())
    loudness = clip.dBFS if isfinite(clip.dBFS) else -60.0
    loudness_score = max(0.0, 1.0 - abs(loudness + 18.0) / 18.0)
    duration_score = min(clip_length_ms / 6000.0, 1.0)
    word_score = min(word_count / 12.0, 1.0)
    clipping_penalty = 0.4 if clip.max_dBFS > -0.5 else 0.0
    return (speech_ratio * 3.0) + loudness_score + duration_score + word_score - clipping_penalty


def _cleanup_voice_sample_clip(clip: AudioSegment) -> AudioSegment:
    """Apply basic cleanup to a speaker reference clip before cloning."""
    if len(clip) <= 0:
        return clip

    cleaned_clip = effects.high_pass_filter(clip, cutoff=80)
    cleaned_clip = _trim_clip_silence(cleaned_clip)
    if len(cleaned_clip) <= 0:
        return cleaned_clip
    cleaned_clip = effects.compress_dynamic_range(cleaned_clip, threshold=-18.0, ratio=3.0, attack=5.0, release=50.0)
    cleaned_clip = effects.normalize(cleaned_clip, headroom=1.0)
    return cleaned_clip.fade_in(15).fade_out(15)


def _trim_clip_silence(clip: AudioSegment, padding_ms: int = 80) -> AudioSegment:
    """Trim long silent regions from a reference clip while preserving short speech boundaries."""
    silence_threshold = _voice_sample_silence_threshold(clip)
    non_silent_ranges = silence.detect_nonsilent(clip, min_silence_len=250, silence_thresh=silence_threshold)
    if not non_silent_ranges:
        return clip

    trimmed_clip = AudioSegment.empty()
    for start_ms, end_ms in non_silent_ranges:
        trimmed_clip += clip[max(0, start_ms - padding_ms) : min(len(clip), end_ms + padding_ms)]
    return trimmed_clip


def _voice_sample_silence_threshold(clip: AudioSegment) -> float:
    """Return a silence threshold appropriate for the current clip loudness."""
    clip_dbfs = clip.dBFS if isfinite(clip.dBFS) else -45.0
    return max(-50.0, clip_dbfs - 16.0)


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