"""
rvc_batch - Simplified high-level RVC voice conversion API.

Usage:
    from rvc_batch import RVC

    vc = RVC("path/to/model.pth", f0_method="rmvpe")
    vc.convert("input.wav", "output.wav", pitch_change=2)

    # Batch mode
    vc.convert("input_folder/", "output_folder/", pitch_change=2)
"""

import os
import torch

from rvc_batch.infer.infer import (
    Config,
    load_hubert,
    get_vc,
    rvc_infer,
    rvc_infer_batch,
)
from rvc_batch.utils import check_predictors, check_embedders
from rvc_batch.config.variable import MODELS_DIR


class RVC:
    """Simple RVC voice conversion wrapper.

    Load a voice model and convert audio files with a clean 2-method API.
    Supports single-file and batch (folder) conversion, with optional
    pitch shifting, autotune, and FAISS index-based feature matching.

    Args:
        model_path: Path to a .pth voice model file.
        index_path: Optional path to a .index FAISS file.
        f0_method: Pitch extraction method (default "rmvpe").
        device: Device to run on (default auto-detect).
        is_half: Use half precision (default auto-detect).
        models_dir: Directory where predictor/embedder models are stored.
    """

    def __init__(self, model_path, index_path="", f0_method="rmvpe",
                 device=None, is_half=None, models_dir=None):
        if models_dir:
            os.environ["RVC_MODELS_DIR"] = models_dir
            from rvc_batch.config import variable as _var
            _var.MODELS_DIR = models_dir

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.index_path = index_path
        self.f0_method = f0_method

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                is_half = True if is_half is None else is_half
            elif torch.backends.mps.is_available():
                device = "mps"
                is_half = True if is_half is None else is_half
            else:
                device = "cpu"
                is_half = False if is_half is None else is_half
        else:
            is_half = is_half if is_half is not None else True

        # Download required models
        check_embedders()
        check_predictors(f0_method)

        config = Config(device, is_half)
        hubert_path = os.path.join(MODELS_DIR, "hubert_base.pt")
        self.hubert_model = load_hubert(device, is_half, hubert_path)

        self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(
            device, is_half, config, model_path
        )

        self.device = device
        self.is_half = is_half

    def convert(self, input_path, output_path, pitch_change=0,
                index_rate=0.75, filter_radius=3, rms_mix_rate=0.25,
                protect=0.33, f0_autotune=False, f0_autotune_strength=1.0,
                autotune_key=None, autotune_scale=None):
        """Convert audio file(s).

        If input_path is a folder, all supported audio files inside are
        converted and saved to output_path (also a folder).

        Returns a dict with 'processed', 'failed', 'skipped' counts
        for batch mode, or None for single-file mode.
        """
        if os.path.isdir(input_path):
            return rvc_infer_batch(
                index_path=self.index_path,
                index_rate=index_rate,
                input_path=input_path,
                output_path=output_path,
                pitch_change=pitch_change,
                f0_method=self.f0_method,
                cpt=self.cpt, version=self.version,
                net_g=self.net_g,
                filter_radius=filter_radius,
                tgt_sr=self.tgt_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                crepe_hop_length=128,
                vc=self.vc,
                hubert_model=self.hubert_model,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                autotune_key=autotune_key,
                autotune_scale=autotune_scale,
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        rvc_infer(
            index_path=self.index_path,
            index_rate=index_rate,
            input_path=input_path,
            output_path=output_path,
            pitch_change=pitch_change,
            f0_method=self.f0_method,
            cpt=self.cpt, version=self.version,
            net_g=self.net_g,
            filter_radius=filter_radius,
            tgt_sr=self.tgt_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            crepe_hop_length=128,
            vc=self.vc,
            hubert_model=self.hubert_model,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            autotune_key=autotune_key,
            autotune_scale=autotune_scale,
        )
        return None
