"""Setuptools compatibility entrypoint with Python version guard."""

from __future__ import annotations

import sys

from setuptools import setup


if not ((3, 10) <= sys.version_info < (3, 12)):
    raise RuntimeError(
        "DubbSystem requires Python 3.10 or 3.11. "
        "The Coqui TTS dependency used for XTTS voice cloning does not currently support Python 3.12."
    )


setup()