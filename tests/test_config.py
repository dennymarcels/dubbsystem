"""Tests for configuration helpers."""

from __future__ import annotations

from pathlib import Path

from dubb.schemas import DubbingConfig


def test_default_output_path_uses_dubbed_suffix() -> None:
    """Ensure the default output path is derived from the input filename."""
    config = DubbingConfig(input_path=Path("/tmp/example.mp4"))
    assert config.resolved_output_path == Path("/tmp/_dubbedexample.mp4")


def test_explicit_output_path_is_preserved() -> None:
    """Ensure an explicit output path is not rewritten."""
    config = DubbingConfig(
        input_path=Path("/tmp/example.mp4"),
        output_path=Path("/tmp/custom.mp4"),
    )
    assert config.resolved_output_path == Path("/tmp/custom.mp4")


def test_default_target_language_uses_american_english() -> None:
    """Ensure the default target language is American English."""
    config = DubbingConfig(input_path=Path("/tmp/example.mp4"))
    assert config.target_language == "en-us"