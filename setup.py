"""Setuptools compatibility entrypoint with Python version guard."""

from __future__ import annotations

import sys

from setuptools import setup


if not ((3, 10) <= sys.version_info < (3, 12)):
    raise RuntimeError(
        "DubbSystem requires Python 3.10 or 3.11. "
        "This is currently limited by the pinned transformers<4.56 requirement needed for XTTS v2 "
        "compatibility, not by the coqui-tts package itself, which now supports newer Python versions."
    )


setup()