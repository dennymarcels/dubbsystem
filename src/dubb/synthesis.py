"""Speech synthesis utilities using Coqui XTTS."""

from __future__ import annotations

from pathlib import Path

import torch
from TTS.api import TTS


class VoiceCloner:
    """Synthesize speech using a voice reference sample."""

    def __init__(self, model_name: str, device: str) -> None:
        """Load the TTS model onto the configured device."""
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