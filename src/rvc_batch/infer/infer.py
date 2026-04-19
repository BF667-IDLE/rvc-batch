import os
import time
from multiprocessing import cpu_count
from pathlib import Path

import torch
from scipy.io import wavfile

from rvc_batch.synth.fairseq import load_model as load_hubert_model
from rvc_batch.synth.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc_batch.utils import load_audio
from rvc_batch.infer.pipeline import VC
from rvc_batch.config.variable import SUPPORTED_EXTENSIONS, MODELS_DIR


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device, is_half, model_path):
    hubert = load_hubert_model(model_path)
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu')
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def _ensure_hubert_loaded(hubert_model, device, is_half):
    """Auto-load HuBERT base model if not already provided."""
    if hubert_model is not None:
        return hubert_model

    from rvc_batch.utils import check_embedders
    from rvc_batch.config.variable import MODELS_DIR

    print("HuBERT model not provided — auto-loading HuBERT base model...")
    check_embedders()
    hubert_path = os.path.join(MODELS_DIR, "hubert_base.pt")
    return load_hubert(device, is_half, hubert_path)


def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, f0_autotune=False, f0_autotune_strength=1.0, autotune_key=None, autotune_scale=None):
    from rvc_batch.config.variable import SAMPLE_RATE
    hubert_model = _ensure_hubert_loaded(hubert_model, vc.device, vc.is_half)
    audio = load_audio(input_path, SAMPLE_RATE)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, autotune_key=autotune_key, autotune_scale=autotune_scale)
    wavfile.write(output_path, tgt_sr, audio_opt)


def rvc_infer_batch(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, f0_autotune=False, f0_autotune_strength=1.0, autotune_key=None, autotune_scale=None):
    """Batch inference: process a single file or all audio files in a folder.

    Args:
        input_path: Path to a single audio file or a folder containing audio files.
        output_path: Path to a single output file (if input is a file) or an output folder (if input is a folder).
        All other arguments are the same as rvc_infer.

    Returns:
        dict: Summary with keys 'processed', 'failed', 'skipped', 'total_time'.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    # Detect if input is a single file or a folder
    if os.path.isfile(input_path):
        # Single file mode - just delegate to rvc_infer
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        rvc_infer(
            index_path, index_rate, input_path, output_path,
            pitch_change, f0_method, cpt, version, net_g,
            filter_radius, tgt_sr, rms_mix_rate, protect,
            crepe_hop_length, vc, hubert_model,
            f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength,
            autotune_key=autotune_key, autotune_scale=autotune_scale,
        )
        return {"processed": 1, "failed": 0, "skipped": 0, "total_time": 0}

    # Folder mode
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Collect all supported audio files
    audio_files = []
    for entry in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, entry)
        if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(full_path)

    if not audio_files:
        print(f"No supported audio files found in: {input_path}")
        return {"processed": 0, "failed": 0, "skipped": 0, "total_time": 0}

    print(f"Found {len(audio_files)} audio file(s) in: {input_path}")
    summary = {"processed": 0, "failed": 0, "skipped": 0, "total_time": 0}
    start_time = time.time()

    for i, audio_file in enumerate(audio_files, 1):
        # Build output file path: preserve filename, change extension to .wav
        filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
        out_file = os.path.join(output_path, filename)

        # Skip if output already exists
        if os.path.exists(out_file):
            print(f"[{i}/{len(audio_files)}] Skipping (already exists): {os.path.basename(audio_file)}")
            summary["skipped"] += 1
            continue

        print(f"[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}", end=" ", flush=True)
        file_start = time.time()

        try:
            rvc_infer(
                index_path, index_rate, audio_file, out_file,
                pitch_change, f0_method, cpt, version, net_g,
                filter_radius, tgt_sr, rms_mix_rate, protect,
                crepe_hop_length, vc, hubert_model,
                f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength,
                autotune_key=autotune_key, autotune_scale=autotune_scale,
            )
            elapsed = time.time() - file_start
            print(f"-> Done ({elapsed:.2f}s)")
            summary["processed"] += 1
        except Exception as e:
            elapsed = time.time() - file_start
            print(f"-> FAILED ({elapsed:.2f}s): {e}")
            summary["failed"] += 1

    summary["total_time"] = time.time() - start_time
    print(
        f"\nBatch complete: {summary['processed']} processed, "
        f"{summary['failed']} failed, {summary['skipped']} skipped, "
        f"total time: {summary['total_time']:.2f}s"
    )
    return summary
