"""Translation utilities using Hugging Face models."""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dubb.schemas import Segment


class Translator:
    """Translate segment text with a sequence-to-sequence model."""

    _NLLB_LANGUAGE_MAP: dict[str, str] = {
        "ar": "arb_Arab",
        "de": "deu_Latn",
        "en": "eng_Latn",
        "es": "spa_Latn",
        "fr": "fra_Latn",
        "hi": "hin_Deva",
        "it": "ita_Latn",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "nl": "nld_Latn",
        "pl": "pol_Latn",
        "pt": "por_Latn",
        "ru": "rus_Cyrl",
        "tr": "tur_Latn",
        "uk": "ukr_Cyrl",
        "zh": "zho_Hans",
    }

    def __init__(self, model_name: str, device: str, source_language: str, target_language: str) -> None:
        """Load the tokenizer and model for translation."""
        self._model_name = model_name
        self._source_language = source_language
        self._target_language = target_language
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._torch_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model.to(self._torch_device)

    def translate_segments(self, segments: Iterable[Segment]) -> list[Segment]:
        """Translate segment text and preserve timestamps."""
        translated_segments: list[Segment] = []
        for segment in segments:
            translated_text = self._translate_text(segment.text)
            translated_segments.append(segment.model_copy(update={"translated_text": translated_text}))
        return translated_segments

    def _translate_text(self, text: str) -> str:
        """Translate a single text fragment."""
        if "nllb" in self._model_name.lower():
            source_code = self._normalize_language_code(self._source_language)
            target_code = self._normalize_language_code(self._target_language)
            self._tokenizer.src_lang = source_code
            encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
            encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
            generated = self._model.generate(
                **encoded,
                max_new_tokens=256,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_code),
            )
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
        generated = self._model.generate(**encoded, max_new_tokens=256)
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _normalize_language_code(self, language_code: str) -> str:
        """Map ISO-like codes to the NLLB token space when required."""
        return self._NLLB_LANGUAGE_MAP.get(language_code, language_code)