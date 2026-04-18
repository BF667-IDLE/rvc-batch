"""rvc-batch CLI - Command-line interface for batch voice conversion."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="rvc-batch",
        description="RVC-Batch: High-quality voice conversion with batch processing support.",
    )
    parser.add_argument("input", help="Input audio file or folder")
    parser.add_argument("output", help="Output audio file or folder")
    parser.add_argument("-m", "--model", required=True, help="Path to .pth voice model")
    parser.add_argument("-i", "--index", default="", help="Path to .index FAISS file")
    parser.add_argument("-p", "--pitch", type=int, default=0, help="Pitch shift in semitones (default: 0)")
    parser.add_argument("-f", "--f0-method", default="rmvpe",
                        choices=["rmvpe", "crepe-full", "crepe-tiny", "crepe-small",
                                 "crepe-medium", "crepe-large", "mangio-crepe-full",
                                 "mangio-crepe-tiny", "mangio-crepe-small",
                                 "mangio-crepe-medium", "mangio-crepe-large",
                                 "fcpe", "fcpe-legacy", "rmvpe-legacy", "swipe",
                                 "pm", "harvest", "dio", "yin", "pyin", "djcm"],
                        help="Pitch extraction method (default: rmvpe)")
    parser.add_argument("--index-rate", type=float, default=0.75,
                        help="FAISS index influence rate 0-1 (default: 0.75)")
    parser.add_argument("--filter-radius", type=int, default=3,
                        help="Median filter radius (default: 3)")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25,
                        help="RMS mix rate 0-1 (default: 0.25)")
    parser.add_argument("--protect", type=float, default=0.33,
                        help="Protect voiceless consonants 0-0.5 (default: 0.33)")
    parser.add_argument("--autotune", action="store_true", help="Enable pitch autotune")
    parser.add_argument("--autotune-strength", type=float, default=1.0,
                        help="Autotune strength 0-1 (default: 1.0)")
    parser.add_argument("--autotune-key", default=None,
                        help="Musical key (e.g. C, F#)")
    parser.add_argument("--autotune-scale", default=None,
                        help="Scale name (e.g. major, minor, blues)")
    parser.add_argument("--models-dir", default=None,
                        help="Directory for predictor/embedder models")
    parser.add_argument("--device", default=None, help="Device: cuda, mps, cpu")
    args = parser.parse_args()

    from rvc_batch import RVC

    print(f"Loading model: {args.model}")
    vc = RVC(
        model_path=args.model,
        index_path=args.index,
        f0_method=args.f0_method,
        device=args.device,
        models_dir=args.models_dir,
    )

    print(f"Converting: {args.input} -> {args.output}")
    result = vc.convert(
        input_path=args.input,
        output_path=args.output,
        pitch_change=args.pitch,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        f0_autotune=args.autotune,
        f0_autotune_strength=args.autotune_strength,
        autotune_key=args.autotune_key,
        autotune_scale=args.autotune_scale,
    )

    if result is not None:
        print(f"Batch complete: {result['processed']} processed, "
              f"{result['failed']} failed, {result['skipped']} skipped")
    else:
        print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
