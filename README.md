# RVC-Batch

Simplified RVC (Retrieval-based Voice Conversion) with batch processing, autotune, and pip-installable packaging.

## Install

```bash
pip install git+https://github.com/BF667-IDLE/rvc-batch.git
```

Or install from source:

```bash
git clone https://github.com/BF667-IDLE/rvc-batch.git
cd rvc-batch
pip install -e ".[gui]"
```

## Quick Start (Python API)

```python
from rvc_batch import RVC

# Load a voice model
vc = RVC("path/to/model.pth", f0_method="rmvpe")

# Single file conversion
vc.convert("input.wav", "output.wav", pitch_change=2)

# Batch folder conversion
vc.convert("input_folder/", "output_folder/", pitch_change=2)
```

## CLI Usage

```bash
# Single file
rvc-batch input.wav output.wav -m model.pth -p 2

# Batch processing
rvc-batch input_folder/ output_folder/ -m model.pth --f0-method rmvpe

# With index and autotune
rvc-batch input.wav output.wav -m model.pth -i model.index --autotune --autotune-key C --autotune-scale minor
```

## Gradio Web UI

```bash
pip install rvc-batch[gui]
rvc-batch-gui  # or: python app.py
```

## Features

- **Batch processing**: Convert entire folders of audio files at once
- **20+ pitch methods**: RMVPE, CREPE (all sizes), FCPE, SWIPE, PM, YIN, DJCM, and more
- **Autotune**: Snap pitch to musical keys and scales
- **FAISS index**: Optional feature matching for improved voice similarity
- **Configurable**: Pitch shift, filter radius, RMS mixing, voiceless consonant protection
- **PyPI installable**: Clean package with `pyproject.toml` and console script entry point

## Model Format

Uses standard RVC v2 `.pth` model files with optional `.index` FAISS files. Place models in a directory structure like:

```
models/
  my_voice/
    model.pth
    model.index  (optional)
```

Set `RVC_MODELS_DIR` environment variable to customize the models directory.
