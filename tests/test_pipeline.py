"""Tests for pipeline validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dubb.pipeline import DubbingPipeline
from dubb.schemas import DubbingConfig
from dubb.synthesis import normalize_xtts_language
from dubb.translation import Translator


def test_pipeline_rejects_missing_input_file(tmp_path: Path) -> None:
    """Ensure a missing input file raises a clear error."""
    pipeline = DubbingPipeline(DubbingConfig(input_path=tmp_path / "missing.mp4"))
    with pytest.raises(FileNotFoundError):
        pipeline.run()


def test_pipeline_rejects_non_mp4_input(tmp_path: Path) -> None:
    """Ensure only MP4 files are accepted."""
    input_file = tmp_path / "clip.mov"
    input_file.write_text("placeholder", encoding="utf-8")
    pipeline = DubbingPipeline(DubbingConfig(input_path=input_file))
    with pytest.raises(ValueError):
        pipeline.run()


def test_american_english_translation_normalization() -> None:
    """Ensure en-us output is normalized toward American English spelling."""
    normalized_text = Translator._normalize_translated_text(
        "The colour programme was organised in the theatre centre.",
        "en-us",
    )
    assert normalized_text == "The color program was organized in the theater center."


def test_xtts_language_aliases_support_american_english() -> None:
    """Ensure XTTS receives a supported language code for en-us dubbing."""
    assert normalize_xtts_language("en-us") == "en"


def test_prepare_workspace_creates_temp_directory(tmp_path: Path) -> None:
    """Ensure the staged workflow prepares a stable artifact directory."""
    input_file = tmp_path / "clip.mp4"
    input_file.write_text("placeholder", encoding="utf-8")
    pipeline = DubbingPipeline(DubbingConfig(input_path=input_file))

    work_dir = pipeline.prepare_workspace()

    assert work_dir == tmp_path / ".dubb_tmp" / "clip"
    assert work_dir.exists()


def test_run_executes_pipeline_steps_in_sequence(tmp_path: Path) -> None:
    """Ensure the whole-process entrypoint calls the staged steps in order."""
    input_file = tmp_path / "clip.mp4"
    input_file.write_text("placeholder", encoding="utf-8")

    class RecordingPipeline(DubbingPipeline):
        def __init__(self, config: DubbingConfig) -> None:
            super().__init__(config)
            self.calls: list[str] = []

        def prepare_workspace(self) -> Path:
            self.calls.append("prepare_workspace")
            return self._config.temp_dir

        def extract_source_audio(self) -> Path:
            self.calls.append("extract_source_audio")
            return self._config.temp_dir / "source.wav"

        def create_speaker_sample(self, source_audio: Path) -> Path:
            self.calls.append(f"create_speaker_sample:{source_audio.name}")
            return self._config.temp_dir / "speaker_sample.wav"

        def transcribe_source_audio(self, source_audio: Path):
            self.calls.append(f"transcribe_source_audio:{source_audio.name}")

            class TranscriptResult:
                segments = []
                source_language = "es"

            return TranscriptResult()

        def translate_segments(self, segments, source_language: str):
            self.calls.append(f"translate_segments:{source_language}")
            return []

        def prepare_synthesis_chunks(self, translated_segments):
            self.calls.append("prepare_synthesis_chunks")
            return []

        def synthesize_chunks(self, chunks, speaker_sample: Path):
            self.calls.append(f"synthesize_chunks:{speaker_sample.name}")
            return []

        def compose_dubbed_audio(self, segments):
            self.calls.append("compose_dubbed_audio")
            return self._config.temp_dir / "dubbed_track.wav"

        def mux_dubbed_video(self, dubbed_audio: Path) -> Path:
            self.calls.append(f"mux_dubbed_video:{dubbed_audio.name}")
            return self._config.resolved_output_path

    pipeline = RecordingPipeline(DubbingConfig(input_path=input_file))

    output_path = pipeline.run()

    assert output_path == input_file.with_name("_dubbedclip.mp4")
    assert pipeline.calls == [
        "prepare_workspace",
        "extract_source_audio",
        "create_speaker_sample:source.wav",
        "transcribe_source_audio:source.wav",
        "translate_segments:es",
        "prepare_synthesis_chunks",
        "synthesize_chunks:speaker_sample.wav",
        "compose_dubbed_audio",
        "mux_dubbed_video:dubbed_track.wav",
    ]