import os
import time
import torch
import gradio as gr

from main.infer.infer import Config, load_hubert, get_vc, rvc_infer
from main.utils import check_predictors, check_embedders, load_audio

# ─── Globals ───
hubert_model = None
cpt = version = net_g = tgt_sr = vc = None

F0_METHODS = [
    "rmvpe", "crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large",
    "fcpe", "fcpe-legacy", "swipe", "pm", "harvest", "dio", "yin", "pyin", "djcm",
]


def load_models(model_path, index_path, f0_method):
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


def inference(audio, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect):
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

    t0 = time.time()
    rvc_infer(
        index_path="", index_rate=index_rate,
        input_path=input_path, output_path=output_path,
        pitch_change=pitch_change, f0_method=f0_method,
        cpt=cpt, version=version, net_g=net_g,
        filter_radius=filter_radius, tgt_sr=tgt_sr,
        rms_mix_rate=rms_mix_rate, protect=protect,
        crepe_hop_length=128, vc=vc, hubert_model=hubert_model,
    )
    elapsed = time.time() - t0

    from scipy.io import wavfile as wf
    sr, data = wf.read(output_path)
    info = f"Done in {elapsed:.1f}s | {version} | {tgt_sr}Hz | {f0_method}"
    return (tgt_sr, data), info


with gr.Blocks(title="RVC-Batch", analytics_enabled=False) as demo:
    gr.Markdown("# RVC-Batch Voice Conversion")

    with gr.Group():
        gr.Markdown("### 1. Load Voice Model")
        with gr.Row():
            model_path = gr.Textbox(label="Model path (.pth)", placeholder="models/your_model.pth", scale=3)
            index_path = gr.Textbox(label="Index path (.index)", placeholder="models/your_index.index", scale=3)
        with gr.Row():
            f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="F0 method")
            btn_load = gr.Button("Load Model", variant="primary")
        load_status = gr.Textbox(label="Status", interactive=False)

    with gr.Group():
        gr.Markdown("### 2. Inference")
        audio_in = gr.Audio(label="Input audio", type="numpy")
        with gr.Row():
            pitch_change = gr.Slider(-12, 12, value=0, step=1, label="Pitch")
            index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index rate")
            filter_radius = gr.Slider(0, 7, value=3, step=1, label="Filter radius")
            rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="RMS mix rate")
            protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect")
        btn_infer = gr.Button("Convert", variant="primary")
        audio_out = gr.Audio(label="Output audio", type="numpy")
        infer_info = gr.Textbox(label="Info", interactive=False)

    btn_load.click(load_models, [model_path, index_path, f0_method], load_status)
    btn_infer.click(inference,
        [audio_in, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect],
        [audio_out, infer_info],
    )


if __name__ == "__main__":
    demo.launch(share=True)
