"""End-to-end dubbing pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from dubb.schemas import DubbingConfig, Segment

logger = logging.getLogger(__name__)


class DubbingPipeline:
    """Coordinate the full dubbing workflow for a single video."""

    def __init__(self, config: DubbingConfig) -> None:
        """Store runtime configuration."""
        self._config = config

    def run(self) -> Path:
        """Execute the dubbing pipeline and return the output video path."""
        self._validate_inputs()
        logger.info("Validated input video: %s", self._config.input_path)

        import ffmpeg

        from dubb.media import create_voice_sample, extract_audio, fit_audio_to_duration, mux_audio_with_video, overlay_segments
        from dubb.synthesis import VoiceCloner
        from dubb.transcription import transcribe_with_timestamps
        from dubb.translation import Translator

        work_dir = self._config.temp_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using working directory: %s", work_dir)

        logger.info("Extracting source audio at %s Hz", self._config.sample_rate)
        source_audio = extract_audio(
            input_video=self._config.input_path,
            output_audio=work_dir / "source.wav",
            sample_rate=self._config.sample_rate,
        )
        logger.info("Source audio extracted to %s", source_audio)

        logger.info("Creating speaker reference sample (%s seconds)", self._config.voice_sample_seconds)
        speaker_sample = create_voice_sample(
            source_audio=source_audio,
            output_sample=work_dir / "speaker_sample.wav",
            duration_seconds=self._config.voice_sample_seconds,
        )
        logger.info("Speaker reference sample saved to %s", speaker_sample)

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

        logger.info(
            "Translating transcript into %s with model %s",
            self._config.target_language,
            self._config.translation_model,
        )
        translator = Translator(
            model_name=self._config.translation_model,
            device=self._config.device,
            source_language=transcript.source_language,
            target_language=self._config.target_language,
        )
        translated_segments = translator.translate_segments(transcript.segments)
        logger.info("Translation completed for %s segments", len(translated_segments))
        transcript_artifact = self._write_transcript_artifacts(translated_segments, work_dir / "transcript.json")
        logger.info("Transcript artifact written to %s", transcript_artifact)

        logger.info("Loading voice cloning model %s", self._config.tts_model)
        cloner = VoiceCloner(model_name=self._config.tts_model, device=self._config.device)
        synthesized_segments = self._synthesize_segments(translated_segments, speaker_sample, cloner, work_dir)
        logger.info("Synthesized %s aligned speech segments", len(synthesized_segments))

        video_duration = float(ffmpeg.probe(str(self._config.input_path))["format"]["duration"])
        logger.info("Compositing dubbed audio track for %.2f seconds of video", video_duration)
        dubbed_audio = overlay_segments(
            segments=synthesized_segments,
            output_audio=work_dir / "dubbed_track.wav",
            duration_seconds=video_duration,
        )
        logger.info("Dubbed audio track written to %s", dubbed_audio)

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
        segments: list[Segment],
        speaker_sample: Path,
        cloner: VoiceCloner,
        work_dir: Path,
    ) -> list[tuple[Path, float]]:
        """Synthesize and duration-fit translated segments."""
        from dubb.media import fit_audio_to_duration

        aligned_segments: list[tuple[Path, float]] = []
        for index, segment in enumerate(segments):
            if not segment.translated_text:
                logger.warning("Skipping empty translated segment at index %s", index)
                continue
            raw_segment_path = work_dir / "segments" / f"segment_{index:04d}_raw.wav"
            aligned_segment_path = work_dir / "segments" / f"segment_{index:04d}.wav"
            logger.info(
                "Synthesizing segment %s/%s at %.2fs for %.2fs",
                index + 1,
                len(segments),
                segment.start,
                segment.duration,
            )
            cloner.synthesize(
                text=segment.translated_text,
                speaker_sample=speaker_sample,
                language=self._config.target_language,
                output_path=raw_segment_path,
            )
            fit_audio_to_duration(
                source_audio=raw_segment_path,
                output_audio=aligned_segment_path,
                target_duration=max(segment.duration, 0.25),
            )
            aligned_segments.append((aligned_segment_path, segment.start))
        return aligned_segments

    def cleanup(self) -> None:
        """Remove intermediate artifacts for the current input file."""
        if self._config.temp_dir.exists():
            shutil.rmtree(self._config.temp_dir)
            logger.info("Removed temporary directory: %s", self._config.temp_dir)

    def _write_transcript_artifacts(self, segments: list[Segment], output_path: Path) -> Path:
        """Persist timestamped transcript and translation data for inspection."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "translated_text": segment.translated_text,
            }
            for segment in segments
        ]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path