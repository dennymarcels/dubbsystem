"""End-to-end dubbing pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import ffmpeg

from dubb.schemas import DubbingConfig, Segment, SynthesisChunk

logger = logging.getLogger(__name__)


class DubbingPipeline:
    """Coordinate the full dubbing workflow for a single video."""

    def __init__(self, config: DubbingConfig) -> None:
        """Store runtime configuration."""
        self._config = config

    def run(self) -> Path:
        """Execute the dubbing pipeline and return the output video path."""
        self.prepare_workspace()
        source_audio = self.extract_source_audio()
        transcript = self.transcribe_source_audio(source_audio)
        speaker_sample = self.create_speaker_sample(source_audio, transcript.segments)
        translated_segments = self.translate_segments(transcript.segments, transcript.source_language)
        synthesis_chunks = self.prepare_synthesis_chunks(translated_segments)
        synthesized_segments = self.synthesize_chunks(synthesis_chunks, speaker_sample)
        dubbed_audio = self.compose_dubbed_audio(synthesized_segments)
        return self.mux_dubbed_video(dubbed_audio)

    def prepare_workspace(self) -> Path:
        """Validate inputs and create the working directory for inspectable artifacts."""
        self._validate_inputs()
        logger.info("Validated input video: %s", self._config.input_path)
        work_dir = self._config.temp_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using working directory: %s", work_dir)
        return work_dir

    def source_audio_path(self) -> Path:
        """Return the extracted source audio artifact path."""
        return self._config.temp_dir / "source.wav"

    def speaker_sample_path(self) -> Path:
        """Return the speaker sample artifact path."""
        return self._config.temp_dir / "speaker_sample.wav"

    def speaker_sample_selection_path(self) -> Path:
        """Return the speaker sample selection metadata artifact path."""
        return self._config.temp_dir / "speaker_sample.selection.json"

    def transcript_source_path(self) -> Path:
        """Return the raw transcript artifact path."""
        return self._config.temp_dir / "transcript.source.json"

    def transcript_translated_path(self) -> Path:
        """Return the translated transcript artifact path."""
        return self._config.temp_dir / "transcript.translated.json"

    def transcript_compatibility_path(self) -> Path:
        """Return the compatibility transcript artifact path."""
        return self._config.temp_dir / "transcript.json"

    def synthesis_chunks_path(self) -> Path:
        """Return the synthesis chunk artifact path."""
        return self._config.temp_dir / "synthesis_chunks.json"

    def synthesis_manifest_path(self) -> Path:
        """Return the synthesis manifest artifact path."""
        return self._config.temp_dir / "synthesis_manifest.json"

    def dubbed_audio_path(self) -> Path:
        """Return the final dubbed audio artifact path."""
        return self._config.temp_dir / "dubbed_track.wav"

    def extract_source_audio(self) -> Path:
        """Extract the source audio track and persist it for inspection."""
        from dubb.media import extract_audio

        work_dir = self.prepare_workspace()
        logger.info("Extracting source audio at %s Hz", self._config.sample_rate)
        source_audio = extract_audio(
            input_video=self._config.input_path,
            output_audio=work_dir / "source.wav",
            sample_rate=self._config.sample_rate,
        )
        logger.info("Source audio extracted to %s", source_audio)
        return source_audio

    def create_speaker_sample(self, source_audio: Path, transcript_segments: list[Segment] | None = None) -> Path:
        """Create and persist the speaker reference sample used for cloning."""
        from dubb.media import create_voice_sample

        work_dir = self.prepare_workspace()
        logger.info("Creating speaker reference sample (%s seconds)", self._config.voice_sample_seconds)
        if transcript_segments is None and self.transcript_source_path().exists():
            transcript_segments, _ = self.load_source_transcript()
        sample_result = create_voice_sample(
            source_audio=source_audio,
            output_sample=work_dir / "speaker_sample.wav",
            duration_seconds=self._config.voice_sample_seconds,
            segments=transcript_segments,
        )
        self._write_json_artifact(
            self.speaker_sample_selection_path(),
            {
                "target_duration_seconds": self._config.voice_sample_seconds,
                "cleanup": [
                    "high-pass-filter-80hz",
                    "trim-silence",
                    "compress-dynamic-range",
                    "normalize",
                ],
                "selected_segments": sample_result.selection,
            },
        )
        speaker_sample = sample_result.output_path
        logger.info("Speaker reference sample saved to %s", speaker_sample)
        return speaker_sample

    def transcribe_source_audio(self, source_audio: Path) -> Any:
        """Transcribe the source audio and persist raw transcript artifacts."""
        from dubb.transcription import transcribe_with_timestamps

        work_dir = self.prepare_workspace()
        logger.info("Running transcription with Faster Whisper model %s", self._config.transcription_model)
        transcript = transcribe_with_timestamps(
            audio_path=source_audio,
            model_name=self._config.transcription_model,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )
        logger.info(
            "Transcription completed with %s segments; detected source language: %s",
            len(transcript.segments),
            transcript.source_language,
        )
        transcript_payload = {
            "source_language": transcript.source_language,
            "segments": self._serialize_segments(transcript.segments),
        }
        transcript_artifact = self._write_json_artifact(self.transcript_source_path(), transcript_payload)
        logger.info("Raw transcript artifact written to %s", transcript_artifact)
        return transcript

    def translate_segments(self, segments: list[Segment], source_language: str) -> list[Segment]:
        """Translate transcript segments and persist translated transcript artifacts."""
        from dubb.translation import Translator

        work_dir = self.prepare_workspace()
        logger.info(
            "Translating transcript into %s with model %s",
            self._config.target_language,
            self._config.translation_model,
        )
        translator = Translator(
            model_name=self._config.translation_model,
            device=self._config.device,
            source_language=source_language,
            target_language=self._config.target_language,
        )
        translated_segments = translator.translate_segments(segments)
        logger.info("Translation completed for %s segments", len(translated_segments))
        translated_payload = {
            "target_language": self._config.target_language,
            "segments": self._serialize_segments(translated_segments),
        }
        translated_artifact = self._write_json_artifact(self.transcript_translated_path(), translated_payload)
        compatibility_artifact = self._write_transcript_artifacts(translated_segments, self.transcript_compatibility_path())
        logger.info("Translated transcript artifact written to %s", translated_artifact)
        logger.info("Compatibility transcript artifact written to %s", compatibility_artifact)
        return translated_segments

    def prepare_synthesis_chunks(self, translated_segments: list[Segment]) -> list[SynthesisChunk]:
        """Merge translated segments into inspectable synthesis chunks."""
        work_dir = self.prepare_workspace()
        synthesis_chunks = self._build_synthesis_chunks(translated_segments)
        logger.info("Merged %s transcript segments into %s synthesis chunks", len(translated_segments), len(synthesis_chunks))
        chunks_artifact = self._write_json_artifact(
            self.synthesis_chunks_path(),
            [
                {
                    "start": chunk.start,
                    "end": chunk.end,
                    "duration": chunk.duration,
                    "translated_text": chunk.translated_text,
                }
                for chunk in synthesis_chunks
            ],
        )
        logger.info("Synthesis chunk artifact written to %s", chunks_artifact)
        return synthesis_chunks

    def synthesize_chunks(self, chunks: list[SynthesisChunk], speaker_sample: Path) -> list[tuple[Path, float]]:
        """Synthesize chunk audio and persist raw and aligned segment files."""
        from dubb.synthesis import VoiceCloner

        work_dir = self.prepare_workspace()
        logger.info("Loading voice cloning model %s", self._config.tts_model)
        cloner = VoiceCloner(model_name=self._config.tts_model, device=self._config.device)
        synthesized_segments = self._synthesize_segments(chunks, speaker_sample, cloner, work_dir)
        logger.info("Synthesized %s aligned speech chunks", len(synthesized_segments))
        return synthesized_segments

    def compose_dubbed_audio(self, segments: list[tuple[Path, float]]) -> Path:
        """Overlay aligned chunk audio into a single dubbed track file."""
        from dubb.media import overlay_segments

        work_dir = self.prepare_workspace()
        video_duration = float(ffmpeg.probe(str(self._config.input_path))["format"]["duration"])
        logger.info("Compositing dubbed audio track for %.2f seconds of video", video_duration)
        dubbed_audio = overlay_segments(
            segments=segments,
            output_audio=self.dubbed_audio_path(),
            duration_seconds=video_duration,
        )
        logger.info("Dubbed audio track written to %s", dubbed_audio)
        return dubbed_audio

    def mux_dubbed_video(self, dubbed_audio: Path) -> Path:
        """Mux the dubbed track back into the MP4 container."""
        from dubb.media import mux_audio_with_video

        logger.info("Muxing dubbed audio back into MP4 container")
        output_path = mux_audio_with_video(
            input_video=self._config.input_path,
            dubbed_audio=dubbed_audio,
            output_video=self._config.resolved_output_path,
        )
        logger.info("Muxing complete: %s", output_path)
        return output_path

    def _validate_inputs(self) -> None:
        """Validate source file assumptions before running the pipeline."""
        if not self._config.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self._config.input_path}")
        if self._config.input_path.suffix.lower() != ".mp4":
            raise ValueError("Input video must be an MP4 file")

    def _synthesize_segments(
        self,
        chunks: list[SynthesisChunk],
        speaker_sample: Path,
        cloner: VoiceCloner,
        work_dir: Path,
    ) -> list[tuple[Path, float]]:
        """Synthesize and duration-fit merged translated chunks."""
        from dubb.media import normalize_audio_timing

        aligned_segments: list[tuple[Path, float]] = []
        manifest_entries: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if not chunk.translated_text.strip():
                logger.warning("Skipping empty synthesis chunk at index %s", index)
                continue
            raw_segment_path = work_dir / "segments" / f"segment_{index:04d}_raw.wav"
            aligned_segment_path = work_dir / "segments" / f"segment_{index:04d}.wav"
            logger.info(
                "Synthesizing chunk %s/%s at %.2fs for %.2fs",
                index + 1,
                len(chunks),
                chunk.start,
                chunk.duration,
            )
            cloner.synthesize(
                text=chunk.translated_text,
                speaker_sample=speaker_sample,
                language=self._config.target_language,
                output_path=raw_segment_path,
            )
            normalize_audio_timing(
                source_audio=raw_segment_path,
                output_audio=aligned_segment_path,
                target_duration=max(chunk.duration, 0.25),
                min_tempo_factor=self._config.min_tempo_factor,
                max_tempo_factor=self._config.max_tempo_factor,
            )
            aligned_duration = float(ffmpeg.probe(str(aligned_segment_path))["format"]["duration"])
            aligned_segments.append((aligned_segment_path, chunk.start))
            manifest_entries.append(
                {
                    "index": index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "target_duration": chunk.duration,
                    "actual_duration": aligned_duration,
                    "overflow_duration": max(0.0, aligned_duration - chunk.duration),
                    "translated_text": chunk.translated_text,
                    "raw_audio": str(raw_segment_path),
                    "aligned_audio": str(aligned_segment_path),
                }
            )
        manifest_path = self._write_json_artifact(self.synthesis_manifest_path(), manifest_entries)
        logger.info("Synthesis manifest written to %s", manifest_path)
        return aligned_segments

    def load_source_transcript(self) -> tuple[list[Segment], str]:
        """Load the persisted raw transcript artifact."""
        payload = self._read_json_artifact(self.transcript_source_path())
        segments = [
            Segment(
                start=item["start"],
                end=item["end"],
                text=item["text"],
                translated_text=item.get("translated_text"),
            )
            for item in payload["segments"]
        ]
        return segments, payload.get("source_language", "auto")

    def load_translated_segments(self) -> list[Segment]:
        """Load the persisted translated transcript artifact."""
        payload = self._read_json_artifact(self.transcript_translated_path())
        return [
            Segment(
                start=item["start"],
                end=item["end"],
                text=item["text"],
                translated_text=item.get("translated_text"),
            )
            for item in payload["segments"]
        ]

    def load_synthesis_chunks(self) -> list[SynthesisChunk]:
        """Load the persisted synthesis chunk artifact."""
        payload = self._read_json_artifact(self.synthesis_chunks_path())
        return [
            SynthesisChunk(
                start=item["start"],
                end=item["end"],
                translated_text=item["translated_text"],
            )
            for item in payload
        ]

    def load_synthesized_segments(self) -> list[tuple[Path, float]]:
        """Load the persisted synthesis manifest into overlay-ready segment references."""
        payload = self._read_json_artifact(self.synthesis_manifest_path())
        return [(Path(item["aligned_audio"]), float(item["start"])) for item in payload]

    def _build_synthesis_chunks(self, segments: list[Segment]) -> list[SynthesisChunk]:
        """Merge nearby translated segments into larger chunks with smoother pacing."""
        chunks: list[SynthesisChunk] = []
        current_segments: list[Segment] = []

        for segment in segments:
            if not segment.translated_text:
                continue

            if not current_segments:
                current_segments.append(segment)
                continue

            previous_segment = current_segments[-1]
            gap = max(0.0, segment.start - previous_segment.end)
            merged_duration = segment.end - current_segments[0].start
            should_merge = gap <= self._config.merge_gap_threshold and merged_duration <= self._config.max_chunk_duration

            if should_merge:
                current_segments.append(segment)
                continue

            chunks.append(self._segments_to_chunk(current_segments))
            current_segments = [segment]

        if current_segments:
            chunks.append(self._segments_to_chunk(current_segments))

        return chunks

    def _segments_to_chunk(self, segments: list[Segment]) -> SynthesisChunk:
        """Convert a group of nearby translated segments into a single synthesis chunk."""
        translated_text = " ".join(segment.translated_text.strip() for segment in segments if segment.translated_text)
        return SynthesisChunk(
            start=segments[0].start,
            end=segments[-1].end,
            translated_text=translated_text,
        )

    def cleanup(self) -> None:
        """Remove intermediate artifacts for the current input file."""
        if self._config.temp_dir.exists():
            shutil.rmtree(self._config.temp_dir)
            logger.info("Removed temporary directory: %s", self._config.temp_dir)

    def _write_transcript_artifacts(self, segments: list[Segment], output_path: Path) -> Path:
        """Persist timestamped transcript and translation data for inspection."""
        payload = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "translated_text": segment.translated_text,
            }
            for segment in segments
        ]
        return self._write_json_artifact(output_path, payload)

    def _serialize_segments(self, segments: list[Segment]) -> list[dict[str, Any]]:
        """Convert segment models into JSON-serializable dictionaries."""
        return [
            {
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "text": segment.text,
                "translated_text": segment.translated_text,
            }
            for segment in segments
        ]

    def _write_json_artifact(self, output_path: Path, payload: Any) -> Path:
        """Write a JSON artifact to disk for later inspection."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _read_json_artifact(self, input_path: Path) -> Any:
        """Read a JSON artifact from disk."""
        if not input_path.exists():
            raise FileNotFoundError(f"Artifact not found: {input_path}")
        return json.loads(input_path.read_text(encoding="utf-8"))