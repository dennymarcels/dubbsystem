"""Command-line entrypoint for DubbSystem."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from dubb.pipeline import DubbingPipeline
from dubb.schemas import DubbingConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure application logging for the dubbing CLI."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.command()
def dubb(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input MP4 video."),
    output_path: Path | None = typer.Argument(None, help="Optional output MP4 path."),
    target_language: str = typer.Option("en", "--target-language", help="Target dubbing language code."),
    transcription_model: str = typer.Option("large-v3", "--transcription-model", help="Faster Whisper model name."),
    translation_model: str = typer.Option("facebook/nllb-200-1.3B", "--translation-model", help="Hugging Face translation model name."),
    voice_sample_seconds: int = typer.Option(30, "--voice-sample-seconds", min=5, help="Length of source audio used for voice cloning."),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep intermediate working files."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity: DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Dub a source MP4 video into a translated, voice-cloned output MP4."""
    configure_logging(log_level)
    config = DubbingConfig(
        input_path=input_path,
        output_path=output_path,
        target_language=target_language,
        transcription_model=transcription_model,
        translation_model=translation_model,
        voice_sample_seconds=voice_sample_seconds,
    )
    pipeline = DubbingPipeline(config)
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
            logger.info("Temporary working files preserved at %s", config.temp_dir)


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()