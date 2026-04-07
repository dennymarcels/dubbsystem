"""Command-line entrypoint for DubbSystem."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from dubb.pipeline import DubbingPipeline
from dubb.schemas import DubbingConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
steps_app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
logger = logging.getLogger(__name__)

STEP_COMMAND_NAMES = {
    "extract-audio",
    "create-speaker-sample",
    "transcribe",
    "translate",
    "prepare-synthesis-chunks",
    "synthesize",
    "compose-audio",
    "mux-video",
}


def configure_logging(log_level: str) -> None:
    """Configure application logging for the dubbing CLI."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_config(
    input_path: Path,
    output_path: Path | None,
    target_language: str,
    transcription_model: str,
    translation_model: str,
    voice_sample_seconds: int,
) -> DubbingConfig:
    """Construct a pipeline configuration from CLI values."""
    return DubbingConfig(
        input_path=input_path,
        output_path=output_path,
        target_language=target_language,
        transcription_model=transcription_model,
        translation_model=translation_model,
        voice_sample_seconds=voice_sample_seconds,
    )


def build_pipeline(
    input_path: Path,
    output_path: Path | None,
    target_language: str,
    transcription_model: str,
    translation_model: str,
    voice_sample_seconds: int,
    log_level: str,
) -> DubbingPipeline:
    """Create a configured pipeline for either whole-process or staged execution."""
    configure_logging(log_level)
    config = build_config(
        input_path=input_path,
        output_path=output_path,
        target_language=target_language,
        transcription_model=transcription_model,
        translation_model=translation_model,
        voice_sample_seconds=voice_sample_seconds,
    )
    return DubbingPipeline(config)


def require_artifact(path: Path, description: str) -> Path:
    """Ensure a previous-step artifact exists before continuing."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}. Run the previous step first.")
    return path


@app.command()
def dubb(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code, for example en-us, en, it, pt, or fr."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep intermediate working files."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Dub a source MP4 video into a translated, voice-cloned output MP4."""
    pipeline = build_pipeline(
        input_path=input_path,
        output_path=output_path,
        target_language=target_language,
        transcription_model=transcription_model,
        translation_model=translation_model,
        voice_sample_seconds=voice_sample_seconds,
        log_level=log_level,
    )
    logger.info("Starting dubbing job for %s", input_path)
    try:
        output_video = pipeline.run()
        console.print(f"Dubbed video written to: {output_video}")
        logger.info("Dubbing job completed successfully: %s", output_video)
    except Exception:
        logger.exception("Dubbing job failed")
        raise typer.Exit(code=1) from None
    finally:
        if not keep_temp:
            pipeline.cleanup()
            logger.info("Temporary working files removed")
        else:
            logger.info("Temporary working files preserved at %s", pipeline._config.temp_dir)


@steps_app.command("extract-audio")
def extract_audio_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Extract the source audio artifact for inspection."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    source_audio = pipeline.extract_source_audio()
    console.print(f"Extracted audio written to: {source_audio}")


@steps_app.command("create-speaker-sample")
def create_speaker_sample_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Create the speaker sample artifact from the extracted audio."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    source_audio = require_artifact(pipeline.source_audio_path(), "Source audio artifact")
    require_artifact(pipeline.transcript_source_path(), "Transcript artifact")
    speaker_sample = pipeline.create_speaker_sample(source_audio)
    console.print(f"Speaker sample written to: {speaker_sample}")


@steps_app.command("transcribe")
def transcribe_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Transcribe the extracted source audio and persist transcript artifacts."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    source_audio = require_artifact(pipeline.source_audio_path(), "Source audio artifact")
    pipeline.transcribe_source_audio(source_audio)
    console.print(f"Transcript written to: {pipeline.transcript_source_path()}")


@steps_app.command("translate")
def translate_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Translate the raw transcript and persist translated transcript artifacts."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    segments, source_language = pipeline.load_source_transcript()
    pipeline.translate_segments(segments, source_language)
    console.print(f"Translated transcript written to: {pipeline.transcript_translated_path()}")


@steps_app.command("prepare-synthesis-chunks")
def prepare_synthesis_chunks_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Create the merged synthesis chunk artifact from translated transcript segments."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    translated_segments = pipeline.load_translated_segments()
    pipeline.prepare_synthesis_chunks(translated_segments)
    console.print(f"Synthesis chunks written to: {pipeline.synthesis_chunks_path()}")


@steps_app.command("synthesize")
def synthesize_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Generate aligned synthesized chunk audio and persist the synthesis manifest."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    chunks = pipeline.load_synthesis_chunks()
    speaker_sample = require_artifact(pipeline.speaker_sample_path(), "Speaker sample artifact")
    pipeline.synthesize_chunks(chunks, speaker_sample)
    console.print(f"Synthesis manifest written to: {pipeline.synthesis_manifest_path()}")


@steps_app.command("compose-audio")
def compose_audio_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(60, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Overlay aligned segment audio into the final dubbed track artifact."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    segments = pipeline.load_synthesized_segments()
    dubbed_audio = pipeline.compose_dubbed_audio(segments)
    console.print(f"Dubbed audio written to: {dubbed_audio}")


@steps_app.command("mux-video")
def mux_video_step(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en-us", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-3.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(30, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Mux the final dubbed audio artifact back into the MP4 container."""
    pipeline = build_pipeline(input_path, output_path, target_language, transcription_model, translation_model, voice_sample_seconds, log_level)
    dubbed_audio = require_artifact(pipeline.dubbed_audio_path(), "Dubbed audio artifact")
    output_video = pipeline.mux_dubbed_video(dubbed_audio)
    console.print(f"Dubbed video written to: {output_video}")


def main() -> None:
    """Run the CLI application."""
    if len(sys.argv) > 1 and sys.argv[1] in STEP_COMMAND_NAMES:
        steps_app()
        return
    app()


if __name__ == "__main__":
    main()