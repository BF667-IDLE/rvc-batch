import os
import time
import torch
import gradio as gr
import argparse

from main.infer.infer import Config, load_hubert, get_vc, rvc_infer
from main.utils import check_predictors, check_embedders, load_audio
from main.config.variable import F0_ALL_METHODS, AUTOTUNE_KEYS, AUTOTUNE_SCALES

# ─── Globals ───
hubert_model = None
cpt = version = net_g = tgt_sr = vc = None

F0_METHODS = F0_ALL_METHODS
AUTOTUNE_SCALE_NAMES = list(AUTOTUNE_SCALES.keys())

BASE_DIR = os.getcwd()

rvc_models_dir = os.path.join(BASE_DIR, 'models')


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)




def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)




def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]



def update_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)



def load_models(voice_model, f0_method):
    model_path, index_path  = get_rvc_model(voice_model, is_webui)
    global hubert_model, cpt, version, net_g, tgt_sr, vc

    if not model_path:
        raise gr.Error("Please provide a voice model path (.pth)")
    if not os.path.isfile(model_path):
        raise gr.Error(f"Model file not found: {model_path}")

    os.makedirs("models", exist_ok=True)

    if torch.cuda.is_available():
        device, is_half = "cuda:0", True
    elif torch.backends.mps.is_available():
        device, is_half = "mps", True
    else:
        device, is_half = "cpu", True

    # Download HuBERT + predictor if missing
    check_embedders()
    check_predictors(f0_method)

    config = Config(device, is_half)

    hubert_model = load_hubert(device, is_half, "models/hubert_base.pt")
    cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, model_path)

    return f"Loaded: {os.path.basename(model_path)} ({version}, {tgt_sr}Hz)"


def inference(audio, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale):
    global hubert_model, cpt, version, net_g, tgt_sr, vc

    if hubert_model is None:
        raise gr.Error("Load a voice model first (Section 1)")

    if audio is None:
        raise gr.Error("Please upload an audio file")

    # Save uploaded audio to temp
    input_path = os.path.join("output", "_input.wav")
    output_path = os.path.join("output", "_output.wav")
    os.makedirs("output", exist_ok=True)

    from scipy.io import wavfile
    wavfile.write(input_path, audio[0], audio[1])

    # Resolve autotune params
    at_key = autotune_key if autotune_key != "Auto" else None
    at_scale = autotune_scale if f0_autotune else None

    t0 = time.time()
    rvc_infer(
        index_path="", index_rate=index_rate,
        input_path=input_path, output_path=output_path,
        pitch_change=pitch_change, f0_method=f0_method,
        cpt=cpt, version=version, net_g=net_g,
        filter_radius=filter_radius, tgt_sr=tgt_sr,
        rms_mix_rate=rms_mix_rate, protect=protect,
        crepe_hop_length=128, vc=vc, hubert_model=hubert_model,
        f0_autotune=f0_autotune, f0_autotune_strength=autotune_strength,
        autotune_key=at_key, autotune_scale=at_scale,
    )
    elapsed = time.time() - t0

    from scipy.io import wavfile as wf
    sr, data = wf.read(output_path)
    at_info = f" | Autotune: {autotune_key} {autotune_scale}" if f0_autotune else ""
    info = f"Done in {elapsed:.1f}s | {version} | {tgt_sr}Hz | {f0_method}{at_info}"
    return (tgt_sr, data), info

voice_models = get_current_models(rvc_models_dir)

with gr.Blocks(title="RVC-Batch", analytics_enabled=False) as demo:
    gr.Markdown("# RVC-Batch Voice Conversion")

    gr.Markdown("### 1. Load Voice Model")
    with gr.Row():
        rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Models folder "rvc-batch --> rvc_models". After new models are added into this folder, click the refresh button')
        ref_btn = gr.Button('Refresh Models 🔁', variant='primary')
        ref_btn.click(update_list, None, outputs=rvc_model)
    with gr.Row():
        f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="F0 method")
        btn_load = gr.Button("Load Model", variant="primary")
    load_status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### 2. Inference")
    audio_in = gr.Audio(label="Input audio", type="numpy")
    with gr.Row():
        pitch_change = gr.Slider(-12, 12, value=0, step=1, label="Pitch")
        index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index rate")
        filter_radius = gr.Slider(0, 7, value=3, step=1, label="Filter radius")
        rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="RMS mix rate")
        protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect")
    with gr.Row():
        f0_autotune = gr.Checkbox(label="Autotune", value=False)
        autotune_strength = gr.Slider(0, 1, value=1.0, step=0.05, label="Autotune Strength")
        autotune_key = gr.Dropdown(["Auto"] + AUTOTUNE_KEYS, value="Auto", label="Key", interactive=True)
        autotune_scale = gr.Dropdown(AUTOTUNE_SCALE_NAMES, value="major", label="Scale", interactive=True)
    btn_infer = gr.Button("Convert", variant="primary")
    audio_out = gr.Audio(label="Output audio", type="numpy")
    infer_info = gr.Textbox(label="Info", interactive=False)

    btn_load.click(load_models, [voice_model, f0_method], load_status)
    btn_infer.click(inference,
        [audio_in, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale],
        [audio_out, infer_info],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC-Batch Voice Conversion Web UI")
    parser.add_argument("-s", "--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()
    
    demo.launch(share=args.share)
