"""
rvc_batch - Simplified RVC voice conversion library.

Install:
    pip install rvc-batch

Quick start:
    from rvc_batch import RVC

    vc = RVC("model.pth", f0_method="rmvpe")
    vc.convert("input.wav", "output.wav", pitch_change=2)
    vc.convert("input_folder/", "output_folder/")
"""

__version__ = "0.1.0"

from rvc_batch.rvc import RVC

__all__ = ["RVC"]
