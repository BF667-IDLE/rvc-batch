import os
import gc
import sys
import torch
import codecs
import librosa
import requests

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from main.config.variable import SAMPLE_RATE, PREDICTOR_MODEL_DICT, HF_PREDICTOR_BASE, HF_EMBEDDER_BASE

sys.path.append(os.getcwd())


def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif opencl.is_available(): opencl.pytorch_ocl.empty_cache()

def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    output_path = os.path.basename(url) if output_path is None else (os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path)
    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                f.write(chunk)

        return output_path
    else: raise ValueError(response.status_code)

def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join("models", predictors)): 
           HF_download_file(HF_PREDICTOR_BASE + predictors, os.path.join("models", predictors))

    if method in PREDICTOR_MODEL_DICT: download(PREDICTOR_MODEL_DICT[method])

def check_embedders(hubert="hubert_base"):
    if hubert in ["hubert_base"]:
        hubert += ".pt"
        model_path = os.path.join("models", hubert)
        if not os.path.exists(model_path): 
            HF_download_file("".join([HF_EMBEDDER_BASE, "fairseq/", hubert]), model_path)

def load_audio(file, sample_rate=SAMPLE_RATE):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(f"[ERROR] Not found audio: {file}")

        try:
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error reading audio file: {e}")
    
    return audio.flatten()

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0
