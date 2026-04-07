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
apt-get update
apt-get install -y ffmpeg python3.11 python3.11-venv
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e .
```

The notebook and recommended Colab flow create a local Python 3.11 `.venv` inside the cloned repository, then install and run DubbSystem from that environment. This avoids failures when the Colab notebook kernel is on Python 3.12.

Coqui XTTS also requires license confirmation on first model download. If you have reviewed and accepted the applicable Coqui terms, set `COQUI_TOS_AGREED=1` for the dubbing command in Colab to avoid the interactive prompt.

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
dubb /path/to/input.mp4 /path/to/output.mp4 --target-language en-us
```

Override model choices explicitly:

```bash
dubb /path/to/input.mp4 /path/to/output.mp4 \
	--transcription-model large-v3 \
	--translation-model facebook/nllb-200-3.3B \
	--voice-sample-seconds 30
```

Verbose pipeline logging:

```bash
dubb /path/to/input.mp4 --log-level INFO
```

Step-by-step execution is also available from the CLI. Each command writes artifacts into `.dubb_tmp/<video-stem>/` for inspection:

```bash
dubb extract-audio /path/to/input.mp4 /path/to/output.mp4
dubb create-speaker-sample /path/to/input.mp4 /path/to/output.mp4
dubb transcribe /path/to/input.mp4 /path/to/output.mp4
dubb translate /path/to/input.mp4 /path/to/output.mp4
dubb prepare-synthesis-chunks /path/to/input.mp4 /path/to/output.mp4
dubb synthesize /path/to/input.mp4 /path/to/output.mp4
dubb compose-audio /path/to/input.mp4 /path/to/output.mp4
dubb mux-video /path/to/input.mp4 /path/to/output.mp4
```

Important progress messages include input validation, audio extraction, speaker sample creation, transcription segment count, detected source language, translation, per-segment synthesis, muxing, output path, and cleanup status.

## Colab Notebook

A Colab-ready notebook is included at `notebooks/colab_dubb_demo.ipynb`. It installs FFmpeg, creates a Python 3.11 `.venv` when needed, clones the repository, installs the package, lets you upload an MP4, runs the `dubb` CLI from that environment, and previews the dubbed result.

The notebook also includes a staged workflow where each pipeline phase runs separately and leaves behind inspectable files.

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

The step-by-step workflow writes these additional artifacts into `.dubb_tmp/<video-stem>/`:

- `source.wav`
- `speaker_sample.wav`
- `transcript.source.json`
- `transcript.translated.json`
- `synthesis_chunks.json`
- `synthesis_manifest.json`
- `segments/segment_*_raw.wav`
- `segments/segment_*.wav`
- `dubbed_track.wav`

## Notes

- The default target language is American English (`en-us`).
- The default transcription model is Faster Whisper `large-v3`.
- The default translation model is `facebook/nllb-200-3.3B`.
- The default voice cloning model is Coqui XTTS v2.
- The default speaker reference sample length is 30 seconds.
- The full dependency stack is currently supported on Python 3.10 and 3.11, not Python 3.12.
- Locale aliases such as `en-us` are normalized for both translation and XTTS synthesis. For English dubbing, the pipeline also applies a light American spelling normalization pass.
- XTTS currently needs an older `transformers` 4.x release; this project pins `transformers` to the 4.41 line because newer 4.x and 5.x releases break Coqui TTS 0.22.0 imports.
- XTTS voice cloning quality depends strongly on the cleanliness of the reference sample.
- Alignment is segment-based; if you need word-level forced alignment, extend the transcription stage with WhisperX or another aligner.

## Development

```bash
pip install -e .[dev]
pytest
```