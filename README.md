# Speaker Diarization and Urdu ASR

Pipeline for speaker diarization and Urdu automatic speech recognition. Uses DiariZen (BUT-FIT) for diarization, pyannote for speaker embeddings, and a Hugging Face Urdu ASR model for transcription.

## Requirements

- **Python**: 3.9 or higher (3.10+ recommended).
- **CUDA**: 13.x for GPU acceleration (optional; runs on CPU if unavailable).
- **ffmpeg**: Must be on `PATH` for audio extraction and segment cutting. Install separately (e.g. `ffmpeg` package or [ffmpeg.org](https://ffmpeg.org)).
- **Hugging Face**: Some models may require a token; set `HF_TOKEN` or log in via `huggingface-cli login` if you use gated repos.

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA 13

Install PyTorch and torchvision from the CUDA 13.0 wheel index **before** other dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. DiariZen (if not in PyPI)

If `diarizen` is not available from PyPI, install from the official repository:

```bash
pip install git+https://github.com/BUTSpeechFIT/DiariZen.git
```

Adjust to the exact install method documented in the [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) project.

## Configuration

Before running, set the input video path and optional outputs in `speakers_iden.py` (CONFIGURATION section):

| Variable | Description |
|----------|-------------|
| `VIDEO_FILE` | Absolute or relative path to the input video file. **Required.** |
| `OUTPUT_TXT` | Output path for the transcription text file (default: `transcription_final1.txt`). |
| `SPEAKER_DB` | Path to the speaker embeddings database file (default: `speaker_embeddings1.pkl`). Created if missing. |

Do not commit credentials or machine-specific paths. Use environment variables or a local config for production.

## Running

1. Set `VIDEO_FILE` (and optionally `OUTPUT_TXT`, `SPEAKER_DB`) in `speakers_iden.py`.
2. Ensure ffmpeg is on `PATH`.
3. From the project root:

```bash
python speakers_iden.py
```

The script will:

- Extract and normalize audio from the video.
- Run DiariZen diarization in chunks (default 10 minutes per chunk).
- Merge segments, extract embeddings, and cluster speakers.
- Match clusters to the persisted speaker database (or create new speaker IDs).
- Transcribe each segment with the Urdu ASR model.
- Write the transcript to `OUTPUT_TXT` and the updated speaker DB to `SPEAKER_DB`.
- Print speaker statistics and clean up temporary files.

Exit codes: `0` on success; non-zero on fatal errors (e.g. no speaker turns, missing ffmpeg). On SIGINT/SIGTERM the process performs GPU cleanup and exits.

## Output

- **Transcription file** (`OUTPUT_TXT`): Header with model ID, duration, speaker count, and segment count; then one line per segment: `[HH:MM:SS → HH:MM:SS] Speaker_ID: text`.
- **Speaker database** (`SPEAKER_DB`): Pickle file of speaker IDs to embedding lists; reused across runs for consistent speaker labels.

## Dependencies (summary)

- `torch`, `torchvision` (CUDA 13: install from `https://download.pytorch.org/whl/cu130`)
- `numpy`, `scipy`, `scikit-learn`
- `soundfile`, `pydub`
- `pyannote.audio`, `pyannote.core`
- `transformers`
- `diarizen`

See `requirements.txt` for pinned versions. PyTorch must be installed separately with the CUDA 13 index URL above.

## License and model terms

- DiariZen models (e.g. `BUT-FIT/diarizen-wavlm-large-s80-md-v2`) may be under CC BY-NC 4.0 (non-commercial). Check the model card on Hugging Face.
- Urdu ASR and pyannote models have their own licenses; comply with their terms and any gated-access requirements.
