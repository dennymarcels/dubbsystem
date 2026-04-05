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