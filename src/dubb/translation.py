"""Translation utilities using Hugging Face models."""

from __future__ import annotations

import re
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
        "pt-br": "por_Latn",
        "ru": "rus_Cyrl",
        "tr": "tur_Latn",
        "uk": "ukr_Cyrl",
        "zh": "zho_Hans",
        "en-us": "eng_Latn",
        "en_us": "eng_Latn",
    }

    _AMERICAN_ENGLISH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
        (r"\bcolour\b", "color"),
        (r"\bcolours\b", "colors"),
        (r"\bfavour\b", "favor"),
        (r"\bfavourite\b", "favorite"),
        (r"\bfavourites\b", "favorites"),
        (r"\bhonour\b", "honor"),
        (r"\blabour\b", "labor"),
        (r"\bneighbour\b", "neighbor"),
        (r"\bneighbours\b", "neighbors"),
        (r"\borganise\b", "organize"),
        (r"\borganised\b", "organized"),
        (r"\borganising\b", "organizing"),
        (r"\brealise\b", "realize"),
        (r"\brealised\b", "realized"),
        (r"\brealising\b", "realizing"),
        (r"\bapologise\b", "apologize"),
        (r"\bapologised\b", "apologized"),
        (r"\bapologising\b", "apologizing"),
        (r"\btravelling\b", "traveling"),
        (r"\btravelled\b", "traveled"),
        (r"\bcentre\b", "center"),
        (r"\bmetre\b", "meter"),
        (r"\btheatre\b", "theater"),
        (r"\banalogue\b", "analog"),
        (r"\bdefence\b", "defense"),
        (r"\blicence\b", "license"),
        (r"\bprogramme\b", "program"),
        (r"\bgrey\b", "gray"),
    )

    def __init__(self, model_name: str, device: str, source_language: str, target_language: str) -> None:
        """Load the tokenizer and model for translation."""
        self._model_name = model_name
        self._source_language = source_language
        self._target_language = target_language
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._torch_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if self._torch_device == "cuda" else torch.float32
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=model_dtype)
        if getattr(self._model.generation_config, "max_length", None) is not None:
            self._model.generation_config.max_length = None
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
            decoded_text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            return self._normalize_translated_text(decoded_text, self._target_language)
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
        generated = self._model.generate(**encoded, max_new_tokens=256)
        decoded_text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return self._normalize_translated_text(decoded_text, self._target_language)

    def _normalize_language_code(self, language_code: str) -> str:
        """Map ISO-like codes to the NLLB token space when required."""
        normalized_language = language_code.strip().lower()
        return self._NLLB_LANGUAGE_MAP.get(normalized_language, normalized_language)

    @classmethod
    def _normalize_translated_text(cls, text: str, target_language: str) -> str:
        """Apply locale-specific cleanup to translated output."""
        normalized_target = target_language.strip().lower()
        if normalized_target not in {"en-us", "en_us"}:
            return text

        normalized_text = text
        for pattern, replacement in cls._AMERICAN_ENGLISH_REPLACEMENTS:
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        return normalized_text