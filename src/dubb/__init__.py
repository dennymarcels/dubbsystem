"""DubbSystem package."""

from __future__ import annotations

from dubb.schemas import DubbingConfig

__all__ = ["DubbingConfig", "DubbingPipeline"]


def __getattr__(name: str) -> object:
	"""Lazy-load heavy modules on demand."""
	if name == "DubbingPipeline":
		from dubb.pipeline import DubbingPipeline

		return DubbingPipeline
	raise AttributeError(f"module 'dubb' has no attribute {name!r}")