"""Media processing helpers built on FFmpeg and pydub."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, NamedTuple

import ffmpeg
import numpy as np
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
        acoustic_features = _extract_clip_features(clip)
        score = _score_voice_sample_clip(clip, segment.text, acoustic_features)
        candidates.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "score": score,
                "text": segment.text,
                "strategy": "transcript-ranked",
                "embedding": acoustic_features["embedding"],
                "speech_ratio": acoustic_features["speech_ratio"],
                "flatness": acoustic_features["flatness"],
                "centroid_hz": acoustic_features["centroid_hz"],
            }
        )

    candidates = _filter_to_dominant_speaker_candidates(candidates)
    candidates.sort(key=lambda item: (item["score"], item["speech_ratio"]), reverse=True)
    selected: list[dict[str, Any]] = []
    accumulated_duration_ms = 0
    for candidate in candidates:
        if accumulated_duration_ms >= target_duration_ms:
            break
        segment_duration_ms = candidate["end_ms"] - candidate["start_ms"]
        if segment_duration_ms <= 0:
            continue
        selected.append({key: value for key, value in candidate.items() if key != "embedding"})
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
    selected.sort(key=lambda item: item["start_ms"])
    return selected


def _score_voice_sample_clip(clip: AudioSegment, text: str, acoustic_features: dict[str, Any]) -> float:
    """Compute a heuristic score for how suitable a clip is as a voice reference."""
    clip_length_ms = max(len(clip), 1)
    speech_ratio = acoustic_features["speech_ratio"]
    word_count = len(text.split())
    loudness = clip.dBFS if isfinite(clip.dBFS) else -60.0
    loudness_score = max(0.0, 1.0 - abs(loudness + 18.0) / 18.0)
    duration_score = min(clip_length_ms / 6000.0, 1.0)
    word_score = min(word_count / 12.0, 1.0)
    clipping_penalty = 0.4 if clip.max_dBFS > -0.5 else 0.0
    spectral_focus_score = max(0.0, 1.0 - min(acoustic_features["flatness"], 1.0))
    centroid_score = 1.0 if 120.0 <= acoustic_features["centroid_hz"] <= 4_000.0 else 0.5
    return (speech_ratio * 3.0) + loudness_score + duration_score + word_score + spectral_focus_score + centroid_score - clipping_penalty


def _filter_to_dominant_speaker_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only candidates that are acoustically consistent with the dominant speaker cluster."""
    if len(candidates) <= 1:
        return candidates

    similarity_threshold = 0.82
    best_anchor_index = 0
    best_cluster_score = float("-inf")
    for index, candidate in enumerate(candidates):
        cluster_members = [other for other in candidates if _cosine_similarity(candidate["embedding"], other["embedding"]) >= similarity_threshold]
        cluster_score = sum(member["score"] for member in cluster_members)
        if cluster_score > best_cluster_score:
            best_cluster_score = cluster_score
            best_anchor_index = index

    anchor_embedding = candidates[best_anchor_index]["embedding"]
    filtered_candidates = []
    for candidate in candidates:
        similarity = _cosine_similarity(anchor_embedding, candidate["embedding"])
        if similarity >= similarity_threshold:
            filtered_candidates.append({**candidate, "speaker_similarity": similarity})

    return filtered_candidates or candidates


def _extract_clip_features(clip: AudioSegment) -> dict[str, Any]:
    """Extract lightweight acoustic features for speaker consistency ranking."""
    import librosa

    samples = np.array(clip.get_array_of_samples(), dtype=np.float32)
    if clip.channels > 1:
        samples = samples.reshape((-1, clip.channels)).mean(axis=1)
    sample_width_scale = float(1 << (8 * clip.sample_width - 1))
    if sample_width_scale > 0:
        samples = samples / sample_width_scale

    if samples.size == 0:
        embedding = np.zeros(26, dtype=np.float32)
        return {"embedding": embedding, "speech_ratio": 0.0, "flatness": 1.0, "centroid_hz": 0.0}

    silence_threshold = _voice_sample_silence_threshold(clip)
    speech_ranges = silence.detect_nonsilent(clip, min_silence_len=250, silence_thresh=silence_threshold)
    speech_duration_ms = sum(end - start for start, end in speech_ranges)
    speech_ratio = speech_duration_ms / max(len(clip), 1)

    sample_rate = clip.frame_rate
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    spectral_flatness = librosa.feature.spectral_flatness(y=np.maximum(np.abs(samples), 1e-6))

    embedding = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc_delta.mean(axis=1),
        ]
    ).astype(np.float32)
    return {
        "embedding": embedding,
        "speech_ratio": float(speech_ratio),
        "flatness": float(np.mean(spectral_flatness)),
        "centroid_hz": float(np.mean(spectral_centroid)),
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


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
        allow_overflow=False,
    )


def normalize_audio_timing(
    source_audio: Path,
    output_audio: Path,
    target_duration: float,
    min_tempo_factor: float,
    max_tempo_factor: float,
    allow_overflow: bool = True,
) -> Path:
    """Adjust clip pacing within bounded tempo limits, then pad short clips without truncating speech."""
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
    if not allow_overflow and len(clip) > target_duration_ms:
        overflow_ratio = len(clip) / target_duration_ms
        overflow_adjusted_audio = output_audio.with_name(f"{output_audio.stem}_cap{output_audio.suffix}")
        capped_stream = ffmpeg.input(str(tempo_adjusted_audio)).audio
        for factor in _build_atempo_factors(overflow_ratio):
            capped_stream = capped_stream.filter("atempo", factor)
        (
            capped_stream.output(str(overflow_adjusted_audio), ac=1, ar=24_000)
            .overwrite_output()
            .run(quiet=True)
        )
        clip = AudioSegment.from_file(overflow_adjusted_audio)
        overflow_adjusted_audio.unlink(missing_ok=True)

    if len(clip) < target_duration_ms:
        clip += AudioSegment.silent(duration=target_duration_ms - len(clip))
    elif not allow_overflow and len(clip) > target_duration_ms:
        clip = clip[:target_duration_ms]
    clip.export(output_audio, format="wav")
    tempo_adjusted_audio.unlink(missing_ok=True)
    return output_audio


def condense_speech_pauses(
    source_audio: Path,
    output_audio: Path,
    keep_silence_ms: int = 90,
    min_silence_len: int = 220,
) -> Path:
    """Condense long silent pauses in a synthesized clip while preserving speech order."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    clip = AudioSegment.from_file(source_audio)
    silence_threshold = _voice_sample_silence_threshold(clip)
    non_silent_ranges = silence.detect_nonsilent(clip, min_silence_len=min_silence_len, silence_thresh=silence_threshold)
    if not non_silent_ranges:
        clip.export(output_audio, format="wav")
        return output_audio

    condensed_clip = AudioSegment.empty()
    for index, (start_ms, end_ms) in enumerate(non_silent_ranges):
        if index > 0:
            condensed_clip += AudioSegment.silent(duration=keep_silence_ms)
        condensed_clip += clip[start_ms:end_ms]
    condensed_clip.export(output_audio, format="wav")
    return output_audio


def _build_atempo_factors(ratio: float) -> list[float]:
    """Split an atempo ratio into ffmpeg-compatible factors."""
    factors: list[float] = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)
    return factors


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