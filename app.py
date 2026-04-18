import os
import time
import shutil
import tempfile
import torch
import gradio as gr
import argparse

from rvc_batch.infer.infer import Config, load_hubert, get_vc, rvc_infer, rvc_infer_batch
from rvc_batch.utils import check_predictors, check_embedders, load_audio
from rvc_batch.config.variable import F0_ALL_METHODS, AUTOTUNE_KEYS, AUTOTUNE_SCALES, SUPPORTED_EXTENSIONS, MODELS_DIR

# --- Globals ---
hubert_model = None
cpt = version = net_g = tgt_sr = vc = None
current_device = None
current_is_half = None

F0_METHODS = F0_ALL_METHODS
AUTOTUNE_SCALE_NAMES = list(AUTOTUNE_SCALES.keys())


# --- Helper Functions ---


def get_device_config():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda:0", True
    elif torch.backends.mps.is_available():
        return "mps", True
    else:
        return "cpu", False


def get_device_info():
    """Return a human-readable device info string."""
    device, is_half = get_device_config()
    info = f"Device: {device} | Half precision: {is_half}"
    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info = f"Device: {device} ({gpu_name}) | VRAM: {gpu_mem:.1f} GB | Half precision: {is_half}"
    return info


def get_rvc_model_paths(voice_model):
    """Resolve .pth and .index files for a given voice model directory."""
    model_dir = os.path.join(MODELS_DIR, voice_model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pth_path = None
    index_path = None

    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pth":
            pth_path = os.path.join(model_dir, file)
        elif ext == ".index":
            index_path = os.path.join(model_dir, file)

    if pth_path is None:
        raise FileNotFoundError(f"No .pth model file found in {model_dir}")

    return pth_path, index_path or ""


def get_current_models(models_dir):
    """List all subdirectories in the models directory that contain a .pth file."""
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return []

    models_list = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            has_pth = any(f.endswith(".pth") for f in os.listdir(item_path))
            if has_pth:
                models_list.append(item)
    return sorted(models_list)


def update_model_list():
    """Refresh the model dropdown."""
    models_list = get_current_models(MODELS_DIR)
    return gr.update(choices=models_list, value=models_list[0] if models_list else None)


# --- Model Loading ---


def load_models(voice_model, f0_method):
    """Load HuBERT and voice model into memory."""
    global hubert_model, cpt, version, net_g, tgt_sr, vc, current_device, current_is_half

    if not voice_model:
        raise gr.Error("Please select or upload a voice model first.")

    model_path, index_path = get_rvc_model_paths(voice_model)

    os.makedirs(MODELS_DIR, exist_ok=True)

    device, is_half = get_device_config()
    current_device = device
    current_is_half = is_half

    config = Config(device, is_half)

    # Download required models if missing
    check_embedders()
    if f0_method:
        check_predictors(f0_method)

    hubert_path = os.path.join(MODELS_DIR, "hubert_base.pt")
    hubert_model = load_hubert(device, is_half, hubert_path)
    cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, model_path)

    index_info = f" + index" if index_path else ""
    return f"Loaded: {os.path.basename(model_path)} ({version}, {tgt_sr}Hz){index_info} on {device}"


# --- Single Inference ---


def inference(audio, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale):
    """Run single file voice conversion."""
    global hubert_model, cpt, version, net_g, tgt_sr, vc

    if hubert_model is None:
        raise gr.Error("Load a voice model first (Section 1)")

    if audio is None:
        raise gr.Error("Please upload an audio file")

    os.makedirs("output", exist_ok=True)

    # Handle numpy audio tuple (sample_rate, audio_data)
    if isinstance(audio, tuple):
        sr, audio_data = audio
        input_path = os.path.join("output", "_input.wav")
        import soundfile as sf
        sf.write(input_path, audio_data, sr)
    else:
        input_path = audio if isinstance(audio, str) else None
        if input_path is None or not os.path.isfile(input_path):
            raise gr.Error("Invalid audio input")

    output_path = os.path.join("output", "_output.wav")

    # Resolve model paths for index
    _, index_path_val = get_rvc_model_paths(rvc_model.value) if rvc_model.value else ("", "")

    # Resolve autotune params
    at_key = autotune_key if autotune_key != "Auto" else None
    at_scale = autotune_scale if f0_autotune else None

    t0 = time.time()
    rvc_infer(
        index_path=index_path_val, index_rate=index_rate,
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

    import soundfile as sf
    sr_out, data = sf.read(output_path)
    at_info = f" | Autotune: {at_key} {at_scale}" if f0_autotune else ""
    info = f"Done in {elapsed:.1f}s | {version} | {tgt_sr}Hz | {f0_method}{at_info}"
    return (sr_out, data), info


# --- Batch Inference ---


def batch_inference(input_folder, output_folder, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale):
    """Run batch voice conversion on a folder of audio files."""
    global hubert_model, cpt, version, net_g, tgt_sr, vc

    if hubert_model is None:
        raise gr.Error("Load a voice model first (Section 1)")

    if not input_folder or not os.path.isdir(input_folder):
        raise gr.Error("Please provide a valid input folder")

    if not output_folder:
        output_folder = os.path.join("output", "batch_" + str(int(time.time())))

    os.makedirs(output_folder, exist_ok=True)

    _, index_path_val = get_rvc_model_paths(rvc_model.value) if rvc_model.value else ("", "")

    at_key = autotune_key if autotune_key != "Auto" else None
    at_scale = autotune_scale if f0_autotune else None

    t0 = time.time()
    summary = rvc_infer_batch(
        index_path=index_path_val, index_rate=index_rate,
        input_path=input_folder, output_path=output_folder,
        pitch_change=pitch_change, f0_method=f0_method,
        cpt=cpt, version=version, net_g=net_g,
        filter_radius=filter_radius, tgt_sr=tgt_sr,
        rms_mix_rate=rms_mix_rate, protect=protect,
        crepe_hop_length=128, vc=vc, hubert_model=hubert_model,
        f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength,
        autotune_key=at_key, autotune_scale=at_scale,
    )
    elapsed = time.time() - t0

    at_info = f" | Autotune: {at_key} {at_scale}" if f0_autotune else ""
    info = (
        f"Batch complete in {elapsed:.1f}s\n"
        f"Processed: {summary['processed']}\n"
        f"Failed: {summary['failed']}\n"
        f"Skipped: {summary['skipped']}\n"
        f"Total files: {summary['processed'] + summary['failed'] + summary['skipped']}"
        f"{at_info}"
    )
    return info, output_folder


# --- Model Management ---


def upload_model(model_file, model_index, model_name):
    """Upload a .pth model and optionally an .index file to the models directory."""
    if not model_file:
        raise gr.Error("Please provide a .pth model file")

    if not model_name or not model_name.strip():
        model_name = os.path.splitext(os.path.basename(model_file))[0]

    model_dir = os.path.join(MODELS_DIR, model_name.strip())
    os.makedirs(model_dir, exist_ok=True)

    dest_path = os.path.join(model_dir, os.path.basename(model_file))
    shutil.copy2(model_file, dest_path)

    index_dest = ""
    if model_index:
        index_dest = os.path.join(model_dir, os.path.basename(model_index))
        shutil.copy2(model_index, index_dest)

    models_list = get_current_models(MODELS_DIR)
    return (
        gr.update(choices=models_list, value=model_name.strip()),
        f"Model uploaded: {model_name.strip()} ({os.path.basename(model_file)})"
    )


def get_device_status():
    """Display current device information."""
    return get_device_info()


# --- UI ---


voice_models = get_current_models(MODELS_DIR)

with gr.Blocks(title="RVC-Batch", analytics_enabled=False) as demo:
    gr.Markdown("# RVC-Batch Voice Conversion")
    gr.Markdown("A high-quality voice conversion tool with batch processing and autotune support.")

    # Device info
    device_info_box = gr.Textbox(label="Device Info", value=get_device_info(), interactive=False)

    # Section 1: Load Voice Model
    with gr.Accordion("1. Load Voice Model", open=True):
        with gr.Row():
            rvc_model = gr.Dropdown(voice_models, label="Voice Models", info="Select a model from the models folder")
            ref_btn = gr.Button("Refresh Models", variant="primary")

        with gr.Row():
            f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="F0 Method", info="Pitch extraction algorithm (rmvpe recommended)")
            btn_load = gr.Button("Load Model", variant="primary")

        load_status = gr.Textbox(label="Status", interactive=False)

        ref_btn.click(update_model_list, None, rvc_model)
        btn_load.click(load_models, [rvc_model, f0_method], load_status)

    # Section 2: Upload New Model
    with gr.Accordion("2. Upload New Model (Optional)", open=False):
        gr.Markdown("Upload a .pth model file and optionally an .index file to create a new voice model.")
        with gr.Row():
            model_file_input = gr.File(label="Model File (.pth)", file_types=[".pth"])
            model_index_input = gr.File(label="Index File (.index, optional)", file_types=[".index"])
        model_name_input = gr.Textbox(label="Model Name (leave empty to use filename)")
        btn_upload = gr.Button("Upload Model", variant="primary")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

        btn_upload.click(upload_model, [model_file_input, model_index_input, model_name_input], [rvc_model, upload_status])

    # Section 3: Single File Inference
    with gr.Accordion("3. Single File Conversion", open=True):
        audio_in = gr.Audio(label="Input Audio", type="numpy")
        with gr.Row():
            pitch_change = gr.Slider(-12, 12, value=0, step=1, label="Pitch Shift (semitones)")
            index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index Rate")
            filter_radius = gr.Slider(0, 7, value=3, step=1, label="Filter Radius")
        with gr.Row():
            rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="RMS Mix Rate")
            protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect Voiceless Consonants")

        with gr.Row():
            f0_autotune = gr.Checkbox(label="Enable Autotune", value=False)
            autotune_strength = gr.Slider(0, 1, value=1.0, step=0.05, label="Autotune Strength")
            autotune_key = gr.Dropdown(["Auto"] + AUTOTUNE_KEYS, value="Auto", label="Musical Key")
            autotune_scale = gr.Dropdown(AUTOTUNE_SCALE_NAMES, value="major", label="Scale")

        btn_infer = gr.Button("Convert Audio", variant="primary")
        audio_out = gr.Audio(label="Output Audio", type="numpy")
        infer_info = gr.Textbox(label="Info", interactive=False)

        btn_infer.click(
            inference,
            [audio_in, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale],
            [audio_out, infer_info],
        )

    # Section 4: Batch Inference
    with gr.Accordion("4. Batch Folder Processing", open=False):
        gr.Markdown("Process an entire folder of audio files at once. Output files are saved as .wav")
        with gr.Row():
            input_folder = gr.Textbox(label="Input Folder Path", placeholder="path/to/input/folder")
            output_folder = gr.Textbox(label="Output Folder Path", placeholder="path/to/output/folder (auto-generated if empty)")
        btn_batch = gr.Button("Process Folder", variant="primary")
        batch_info = gr.Textbox(label="Batch Result", interactive=False, lines=6)
        batch_output_link = gr.Textbox(label="Output Folder Path", interactive=False)

        btn_batch.click(
            batch_inference,
            [input_folder, output_folder, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, f0_autotune, autotune_strength, autotune_key, autotune_scale],
            [batch_info, batch_output_link],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC-Batch Voice Conversion Web UI")
    parser.add_argument("-s", "--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--port", type=int, default=None, help="Specify port number")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="Specify server address")
    args = parser.parse_args()

    launch_kwargs = {"share": args.share}
    if args.port:
        launch_kwargs["server_port"] = args.port
    if args.server:
        launch_kwargs["server_name"] = args.server

    demo.launch(**launch_kwargs)
