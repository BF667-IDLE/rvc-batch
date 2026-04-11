import os
import codecs
from pathlib import Path

# ─── Paths ───

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MAIN_DIR = os.path.join(BASE_DIR, "main")

# ─── Audio ───

SAMPLE_RATE = 16000
HOP_LENGTH = 160

F0_MIN = 50
F0_MAX = 1100
FRAME_PERIOD = 10

# Butter high-pass filter coefficients (fs=16000, cutoff=48Hz)
from scipy import signal as _signal
BH, AH = _signal.butter(N=5, Wn=48, btype="high", fs=SAMPLE_RATE)

# ─── Supported audio extensions ───

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wma", ".aac", ".webm"}

# ─── Pitch / model constants ───

N_CLASS = 360
N_MELS = 128
PITCH_BINS = 360
CENTS_PER_BIN = 20
MAX_FMAX = 2006
WINDOW_SIZE = 1024
CENTS_MAPPING_BASE = 1997.3794084376191

# ─── F0 methods ───

F0_GENERATOR_METHODS = {
    "pm", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
    "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full",
    "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy", "yin", "pyin", "swipe", "djcm",
}

F0_CREPE_METHOD_MAP = {
    "crepe": "crepe-full",
    "crepe-tiny": "crepe-tiny",
    "crepe-small": "crepe-small",
    "crepe-medium": "crepe-medium",
    "crepe-large": "crepe-large",
    "crepe-full": "crepe-full",
    "mangio-crepe": "mangio-crepe-full",
    "mangio-crepe-tiny": "mangio-crepe-tiny",
    "mangio-crepe-small": "mangio-crepe-small",
    "mangio-crepe-medium": "mangio-crepe-medium",
    "mangio-crepe-large": "mangio-crepe-large",
    "mangio-crepe-full": "mangio-crepe-full",
}

F0_ALL_METHODS = [
    "rmvpe", "crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large",
    "mangio-crepe-full", "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large",
    "fcpe", "fcpe-legacy", "rmvpe-legacy", "swipe", "pm", "harvest", "dio", "yin", "pyin", "djcm",
]

# ─── Autotune ───

AUTOTUNE_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

AUTOTUNE_SCALES = {
    "chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic minor":    [0, 2, 3, 5, 7, 9, 11],
    "pentatonic major": [0, 2, 4, 7, 9],
    "pentatonic minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
}

# Autotune reference frequencies (G1 to C6, 12 semitones per octave)

REF_FREQS = [
    49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41,
    87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83,
    155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63,
    277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16,
    493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61,
    880.00, 932.33, 987.77, 1046.50,
]

# ─── Download URLs (stored as rot13 to avoid raw URLs in source) ───

HF_PREDICTOR_BASE = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
HF_EMBEDDER_BASE = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")

PREDICTOR_MODEL_DICT = {
    **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"),
    **dict.fromkeys(["fcpe"], "fcpe.pt"),
    **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"),
    **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"),
    **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"),
    **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"),
    **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"),
    **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"),
}

# ─── VC pipeline defaults ───

FAISS_SEARCH_K = 8
MAX_INT16 = 32768
HUBERT_V1_OUTPUT_LAYER = 9
HUBERT_V2_OUTPUT_LAYER = 12
FEATURE_SCALE_FACTOR = 2

# ─── DJCM defaults ───

DJCM_WINDOW_LENGTH = 1024
DJCM_N_CLASS = 360
DJCM_SEGMENT_LEN = 5.12

# ─── RMVPE defaults ───

RMVPE_MEL_FMIN = 30
RMVPE_MEL_FMAX = 8000
