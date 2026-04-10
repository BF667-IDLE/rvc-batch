# rvc-batch

A simple, high-quality voice conversion tool focused on ease of use and performance.

## Features

- **Single file inference** - Convert individual audio files with full parameter control
- **Batch folder inference** - Process an entire folder of audio files in one call
- **Multiple F0 methods** - PM, CREPE (tiny/small/medium/large/full), Mangio-CREPE, FCPE, RMVPE, SWIPE, YIN, PYIN, Harvest, DIO, and hybrid combinations
- **RVC v1 & v2 support** - Compatible with both model versions
- **Multiple device support** - CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback
- **Automatic GPU memory config** - Adapts processing parameters based on available VRAM

## Project Structure

```
main/
├── infer/
│   ├── infer.py          # Config, model loading, single & batch inference
│   └── pipeline.py       # VC pipeline (f0 extraction, feature processing)
├── synth/
│   ├── fairseq.py         # HuBERT/fairseq model loading (modular)
│   ├── models.py          # Synthesizer model architectures
│   ├── attentions.py      # Attention mechanisms
│   ├── transforms.py      # Rational quadratic spline transforms
│   ├── modules.py         # WaveNet, ResBlocks, flow modules
│   └── commons.py         # Utility functions for synthesis
├── predictor/
│   ├── generator.py       # Unified F0 generator (modular)
│   ├── rmvpe.py           # RMVPE pitch estimator
│   ├── crepe.py           # CREPE pitch estimator
│   ├── fcpe.py            # FCPE pitch estimator
│   └── swipe.py           # SWIPE pitch estimator
└── utils.py               # Audio loading, auto-download, helpers
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Setup

```python
from infer.infer import Config, load_hubert, get_vc
from utils import check_predictors, check_embedders

# Download required models
check_embedders()           # Downloads HuBERT base model
check_predictors("rmvpe")   # Downloads the f0 predictor model

# Configure device
device = "cuda:0"  # or "mps" or "cpu"
is_half = True     # Use half precision (recommended for GPU)

config = Config(device, is_half)

# Load models
hubert_model = load_hubert(device, is_half, "models/hubert_base.pt")
cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, "models/your_voice_model.pth")
```

### Single File Inference

```python
from infer.infer import rvc_infer

rvc_infer(
    index_path="models/your_index.index",
    index_rate=0.75,
    input_path="input.wav",
    output_path="output.wav",
    pitch_change=0,          # Semitones to shift pitch (-12 to +12)
    f0_method="rmvpe",       # F0 extraction method
    cpt=cpt,
    version=version,
    net_g=net_g,
    filter_radius=3,
    tgt_sr=tgt_sr,
    rms_mix_rate=0.25,
    protect=0.33,
    crepe_hop_length=128,
    vc=vc,
    hubert_model=hubert_model,
)
```

### Batch Folder Inference

```python
from infer.infer import rvc_infer_batch

# Process all audio files in a folder
summary = rvc_infer_batch(
    index_path="models/your_index.index",
    index_rate=0.75,
    input_path="input_folder/",     # Folder containing audio files
    output_path="output_folder/",    # Output folder for converted files
    pitch_change=0,
    f0_method="rmvpe",
    cpt=cpt,
    version=version,
    net_g=net_g,
    filter_radius=3,
    tgt_sr=tgt_sr,
    rms_mix_rate=0.25,
    protect=0.33,
    crepe_hop_length=128,
    vc=vc,
    hubert_model=hubert_model,
)

# summary contains: processed, failed, skipped, total_time
print(f"Processed {summary['processed']} files in {summary['total_time']:.1f}s")
```

**`rvc_infer_batch`** accepts both a single file path or a folder path. When given a folder, it:
- Scans for all supported audio formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.wma`, `.aac`, `.webm`)
- Outputs converted files as `.wav` in the specified output folder
- Skips files that have already been converted (existing output)
- Prints progress and per-file timing
- Returns a summary dictionary

## F0 Methods

| Method | Description |
|--------|-------------|
| `rmvpe` | RMVPE (recommended, fast and accurate) |
| `crepe-full` / `crepe-tiny` / `crepe-small` / `crepe-medium` / `crepe-large` | CREPE variants (balance speed vs quality) |
| `mangio-crepe-full` / etc. | Mangio-CREPE variants |
| `fcpe` / `fcpe-legacy` | FCPE pitch estimator |
| `swipe` | SWIPE algorithm |
| `pm` | Praat parselmouth |
| `harvest` | WORLD harvest |
| `dio` | WORLD DIO |
| `yin` / `pyin` | YIN / pYIN algorithms |
| `hybrid[pm+crepe]` | Hybrid median of multiple methods |

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pitch_change` | Pitch shift in semitones (-12 to +12) | `0` |
| `f0_method` | F0 extraction algorithm | `"rmvpe"` |
| `index_rate` | Feature search influence (0.0 = none, 1.0 = full) | `0.75` |
| `filter_radius` | Median filter radius for pitch smoothing | `3` |
| `rms_mix_rate` | RMS mixing rate (1.0 = full original RMS) | `0.25` |
| `protect` | Voiceless consonant protection (0.0-0.5) | `0.33` |
| `crepe_hop_length` | Hop length for CREPE-based methods | `128` |

## Google Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667-IDLE/rvc-batch/blob/main/rvc_batch_demo.ipynb)

Try it directly in your browser with a free GPU:

1. Click the badge above to open the Colab notebook
2. Go to **Runtime > Change runtime type** and select **GPU** (T4 is free)
3. Run each cell in order - sections are collapsed for cleanliness

The notebook covers the full workflow:
- Clone repo & install dependencies
- Download HuBERT + F0 predictor models automatically
- Upload or link your RVC voice model
- Single file inference with adjustable parameters (sliders)
- Batch folder inference (upload ZIP or multiple files, download results as ZIP)
- In-notebook audio playback of converted files

## License

See [LICENSE](LICENSE) for details.
