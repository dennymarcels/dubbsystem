"""Speech synthesis utilities using Coqui XTTS."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def configure_matplotlib_backend() -> None:
    """Force a safe matplotlib backend before importing Coqui TTS."""
    current_backend = os.environ.get("MPLBACKEND")
    if not current_backend or current_backend.startswith("module://matplotlib_inline"):
        os.environ["MPLBACKEND"] = "Agg"
        logger.info("Using matplotlib backend %s for synthesis", os.environ["MPLBACKEND"])

    try:
        import matplotlib

        matplotlib.use(os.environ["MPLBACKEND"], force=True)
    except Exception:
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib

        matplotlib.use("Agg", force=True)


def configure_torch_checkpoint_loading() -> None:
    """Restore legacy torch.load behavior needed by current Coqui XTTS checkpoints."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    if getattr(torch.load, "_dubb_xtts_compat", False):
        return

    original_torch_load = torch.load

    def patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        """Default to legacy checkpoint loading when libraries omit weights_only."""
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    patched_torch_load._dubb_xtts_compat = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load  # type: ignore[assignment]
    logger.info("Enabled torch checkpoint compatibility mode for XTTS")


class VoiceCloner:
    """Synthesize speech using a voice reference sample."""

    def __init__(self, model_name: str, device: str) -> None:
        """Load the TTS model onto the configured device."""
        configure_matplotlib_backend()
        configure_torch_checkpoint_loading()
        from TTS.api import TTS

        runtime_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._tts = TTS(model_name=model_name).to(runtime_device)

    def synthesize(self, text: str, speaker_sample: Path, language: str, output_path: Path) -> Path:
        """Generate speech audio for translated text."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_sample),
            language=language,
            file_path=str(output_path),
        )
        return output_path