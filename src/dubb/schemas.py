"""Typed schemas for the dubbing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Segment(BaseModel):
    """A timestamped speech segment."""

    start: float
    end: float
    text: str
    translated_text: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Return the duration of the segment in seconds."""
        return max(0.0, self.end - self.start)


class DubbingConfig(BaseModel):
    """Runtime configuration for dubbing."""

    input_path: Path
    output_path: Path | None = None
    target_language: str = Field(default="en")
    transcription_model: str = Field(default="medium")
    translation_model: str = Field(default="facebook/nllb-200-distilled-600M")
    tts_model: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    device: Literal["cpu", "cuda"] = Field(default="cuda")
    compute_type: str = Field(default="float16")
    sample_rate: int = Field(default=24_000)
    voice_sample_seconds: int = Field(default=20)
    temp_dir_name: str = Field(default=".dubb_tmp")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_output_path(self) -> Path:
        """Return the final output path for the dubbed video."""
        if self.output_path is not None:
            return self.output_path
        return self.input_path.with_name(f"_dubbed{self.input_path.name}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_dir(self) -> Path:
        """Return the working directory used for intermediate assets."""
        return self.input_path.parent / self.temp_dir_name / self.input_path.stem