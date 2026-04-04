# DubbSystem

DubbSystem is a Python package for GPU-oriented video dubbing with open source models. It accepts an MP4 video, extracts audio, transcribes it with timestamps, translates the speech to a target language, clones the original speaker voice from a sample, synthesizes dubbed speech, aligns each segment to the source timing, and muxes the result back into an MP4.

## Features

- MP4 input and MP4 output.
- FFmpeg-based audio extraction and video muxing.
- Timestamped transcription with Faster Whisper.
- Translation with Hugging Face sequence-to-sequence models.
- Voice cloning with Coqui XTTS v2.
- Segment-level duration fitting to keep the dubbed track chronologically aligned.
- Installable with `git clone` then `pip install .`.
- CLI entrypoint through `dubb`.

## Installation

### System requirements

- Python 3.10 or 3.11
- FFmpeg available on `PATH`
- CUDA-capable GPU recommended for transcription, translation, and synthesis

Python 3.12 is not currently supported because the Coqui `TTS` dependency used for XTTS voice cloning does not publish compatible distributions for that interpreter line.

### Local install

```bash
git clone <your-repository-url>
cd DubbSystem
pip install .
```

If your local machine defaults to Python 3.12, create a Python 3.10 or 3.11 environment first.

### Google Colab

```bash
git clone <your-repository-url>
cd DubbSystem
pip install .
```

The intended Colab target is a Python 3.10 or 3.11 runtime with GPU enabled.

If Colab does not already provide FFmpeg in your runtime, install it first:

```bash
apt-get update
apt-get install -y ffmpeg
```

## Usage

Default output naming writes the dubbed MP4 next to the source file with an `_dubbed` prefix.

```bash
dubb /path/to/input.mp4
```

Custom output path:

```bash
dubb /path/to/input.mp4 /path/to/output.mp4
```

Optional target language override:

```bash
dubb /path/to/input.mp4 /path/to/output.mp4 --target-language it
```

Verbose pipeline logging:

```bash
dubb /path/to/input.mp4 --log-level INFO
```

Important progress messages include input validation, audio extraction, speaker sample creation, transcription segment count, detected source language, translation, per-segment synthesis, muxing, output path, and cleanup status.

## Colab Notebook

A Colab-ready notebook is included at `notebooks/colab_dubb_demo.ipynb`. It installs FFmpeg, clones the repository, installs the package, lets you upload an MP4, runs the `dubb` CLI, and previews the dubbed result.

## Pipeline

1. Extract source audio as mono 24 kHz WAV.
2. Create a speaker reference sample from the extracted audio.
3. Transcribe the speech with timestamps.
4. Translate text segment by segment.
5. Synthesize translated segments with the cloned voice.
6. Time-fit each segment to the original window.
7. Overlay segments into a single dubbed track.
8. Replace the source audio in the video and export MP4.

Intermediate timestamped transcript data is written to `.dubb_tmp/<video-stem>/transcript.json` while the pipeline runs.

## Notes

- The default transcription model is Faster Whisper `medium`.
- The default translation model is `facebook/nllb-200-distilled-600M`.
- The default voice cloning model is Coqui XTTS v2.
- The full dependency stack is currently supported on Python 3.10 and 3.11, not Python 3.12.
- XTTS voice cloning quality depends strongly on the cleanliness of the reference sample.
- Alignment is segment-based; if you need word-level forced alignment, extend the transcription stage with WhisperX or another aligner.

## Development

```bash
pip install -e .[dev]
pytest
```