#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
import subprocess

from tqdm import tqdm

import whisperx
from lingualign.io import load_and_normalize
from lingualign.lid import init_audio_lid, annotate_segments_language
from lingualign.pipeline import (
    transcribe,
    align,
    diarize,
    write_console_transcript,
    write_latex_screenplay,
)

warnings.filterwarnings("ignore", message=r"You are using `torch.load`")
warnings.filterwarnings("ignore", message=r"No language specified.*")
warnings.filterwarnings("ignore", message=r"audio is shorter than 30s.*")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run multilingual transcript+diarization pipeline on one or more audio files"
    )
    p.add_argument(
        "inputs",
        metavar="AUDIO",
        nargs="+",
        help="path(s) to audio/video file(s) (mp3, m4a, mp4, etc.)",
    )
    p.add_argument(
        "-o", "--output-dir", default="tex",
        help="where to write per-file LaTeX+PDF (default: ./tex/)"
    )
    p.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda"),
        help="where to run WhisperX/SpeechBrain"
    )
    p.add_argument(
        "--compute", default="float32", choices=("float32","float16"),
        help="WhisperX compute_type"
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="WhisperX ASR batch size"
    )
    p.add_argument(
        "--sr", type=int, default=16000,
        help="sampling rate for decode+LID"
    )
    p.add_argument(
        "--margin", type=float, default=0.75,
        help="Zipf margin: if no lang reaches this in text-tie, fall back to audio-LID"
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    # make outputs dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load shared models once
    print(f"▶ Loading WhisperX ASR model ‘medium’ on {args.device}/{args.compute}…")
    try:
        asr = whisperx.load_model("medium", device=args.device, compute_type=args.compute)
    except ValueError as e:
        print(" ⚠️ ", e, "– falling back to cpu/float32")
        asr = whisperx.load_model("medium", device="cpu", compute_type="float32")

    print("▶ Initializing SpeechBrain language-ID model…")
    sb_lid = init_audio_lid(device=args.device)

    for path in args.inputs:
        print(f"\n=== Processing {path} ===")
        # 2) load+normalize audio
        audio, file_sr = load_and_normalize(path, sr=args.sr, mono=True)
        print(f"   → loaded {audio.shape[0]} samples @ {file_sr}Hz")

        # 3) transcribe
        segments, language = transcribe(asr, audio, batch_size=args.batch_size)
        # 4) align
        aligned = align(segments, language, audio, device=args.device)
        # 5) diarize
        dia_segs = diarize(aligned, audio, device=args.device)

        # 6) per-word LID
        annotated = annotate_segments_language(
            dia_segs,
            audio,           # already NP array @ sr
            sr=args.sr,
            model=sb_lid,
            margin=args.margin
        )

        # 7a) console printout
        write_console_transcript(annotated)

        # 7b) LaTeX + PDF
        basename = os.path.splitext(os.path.basename(path))[0]
        tex_path = os.path.join(args.output_dir, f"{basename}.tex")
        pdf_path = os.path.join(args.output_dir, f"{basename}.pdf")
        write_latex_screenplay(
            annotated,
            tex_path,
            title=basename,
        )
        # compile twice
        subprocess.run(["pdflatex", "-interaction=batchmode", os.path.basename(tex_path)],
                       cwd=args.output_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=batchmode", os.path.basename(tex_path)],
                       cwd=args.output_dir, check=True)
        print(f"✔ PDF → {pdf_path}")


if __name__ == "__main__":
    main()
