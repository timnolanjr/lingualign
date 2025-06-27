#!/usr/bin/env python3
"""
lingualign/pipeline.py

Top‐level driver for transcription + segmentation + LID + export.
"""

import os
import argparse
import warnings
import json
from typing import List, Dict, Optional

import whisperx
from tqdm import tqdm

from lingualign.io import load_and_normalize
from lingualign.lid import init_audio_lid, annotate_segments_language
from lingualign.exporters import to_plain, to_markdown, to_srt, to_tex

# Suppress known warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`")
warnings.filterwarnings("ignore", message=r"No language specified.*")
warnings.filterwarnings("ignore", message=r"audio is shorter than 30s.*")


def transcribe_and_align(
    audio:    "np.ndarray",
    sr:       int,
    model:    "whisperx.WhisperModel",
    batch_size: int,
    device:     str
) -> List[Dict]:
    """Run WhisperX ASR -> word‐level alignment."""
    result = model.transcribe(audio, batch_size=batch_size)
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )["segments"]
    return aligned


def diarize(
    aligned: List[Dict],
    audio: "np.ndarray",
    device: str,
    role_map: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Run speaker diarization, assign speakers per word, and apply a role map.
    If only a single speaker is detected, map them to "Narrator" regardless of `role_map`.
    """
    # Diarization and word-level speaker assignment
    diarizer = whisperx.diarize.DiarizationPipeline(device=device)
    turns    = diarizer(audio)
    dia      = whisperx.assign_word_speakers(turns, {"segments": aligned})
    segments = dia["segments"]

    # Collect unique raw speaker labels
    unique_spks = {seg.get("speaker", "UNK") for seg in segments}

    # Determine final mapping: single speaker → Narrator
    if len(unique_spks) == 1:
        only = next(iter(unique_spks))
        final_map = {only: "Narrator"}
    else:
        final_map = role_map or {}

    # Apply mapping to segment and word-level speakers
    for seg in segments:
        raw_spk = seg.get("speaker", "UNK")
        mapped = final_map.get(raw_spk, raw_spk)
        seg["speaker"] = mapped
        for w in seg["words"]:
            # Word-level speaker tags from whisperx.assign_word_speakers
            w_spk = w.get("speaker")
            if w_spk:
                w["speaker"] = final_map.get(w_spk, w_spk)

    return segments


def run_pipeline(
    input_path:    str,
    formats:       List[str],
    output_dir:    str,
    batch_size:    int = 16,
    sr:            int = 16000,
    device:        str = "cpu",
    compute_type:  str = "float32",
    lid_margin:    float = 0.75,
    lid_temp:      float = 5.0,
    role_map:      Optional[str] = None,
) -> None:
    """Main pipeline: decode → ASR+align → diarize → LID → export."""
    os.makedirs(output_dir, exist_ok=True)

    # Load role map JSON if provided
    map_dict: Optional[Dict[str, str]] = None
    if role_map:
        with open(role_map, 'r', encoding='utf-8') as f:
            map_dict = json.load(f)

    print(f"▶ Decoding audio ({input_path!r}) → {sr} Hz mono")
    audio, _ = load_and_normalize(input_path, sr=sr, mono=True)

    print("▶ Loading WhisperX ASR model…")
    asr_model = whisperx.load_model("medium", device=device, compute_type=compute_type)

    print("▶ Transcribing & aligning…")
    aligned = transcribe_and_align(audio, sr, asr_model, batch_size, device)

    print("▶ Diarizing & applying role map…")
    segments = diarize(aligned, audio, device, map_dict)

    print("▶ Initializing audio-LID model…")
    sb_model = init_audio_lid(device=device)

    print("▶ Annotating per-word language…")
    annotated = annotate_segments_language(
        segments,
        audio,
        sr=sr,
        en_lex_path="lexicons/english.txt",
        es_lex_path="lexicons/spanish.txt",
        model=sb_model,
        pad=0.1,
        temp=lid_temp,
        margin=lid_margin,
        fuzzy_cutoff=0.8,
    )

    base = os.path.splitext(os.path.basename(input_path))[0]

    # Export into requested formats
    if "txt" in formats:
        out_txt = os.path.join(output_dir, f"{base}.txt")
        print(f"▶ Writing plain text → {out_txt}")
        to_plain(annotated, out_txt)

    if "md" in formats:
        out_md = os.path.join(output_dir, f"{base}.md")
        print(f"▶ Writing Markdown  → {out_md}")
        to_markdown(annotated, out_md)

    if "srt" in formats:
        out_srt = os.path.join(output_dir, f"{base}.srt")
        print(f"▶ Writing SRT       → {out_srt}")
        to_srt(annotated, out_srt)

    if "tex" in formats:
        tex_dir = os.path.join(output_dir, "tex")
        print(f"▶ Writing LaTeX     → {tex_dir}/{base}.tex")
        to_tex(
            annotated,
            input_path,
            output_dir=tex_dir,
            title=base
        )

    print("✅ Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Run Lingualign transcript pipeline")
    p.add_argument("input", help="path to audio/video file")
    p.add_argument(
        "-f","--formats",
        nargs="+",
        choices=["txt","md","srt","tex"],
        default=["txt","md","srt","tex"],
        help="which output formats to generate"
    )
    p.add_argument(
        "-o","--outdir",
        dest="outdir",
        default="output",
        help="directory to write all outputs into"
    )
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--sr",          type=int,   default=16000)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--compute-type",default="float32")
    p.add_argument("--lid-margin",  type=float, default=0.75,
                   help="Zipf‐sigmoid cutoff before invoking audio LID")
    p.add_argument("--lid-temp",    type=float, default=5.0,
                   help="temperature for audio‐LID softmax")
    p.add_argument(
        "-r","--role-map",
        dest="role_map",
        help="JSON file mapping raw speaker keys to display names",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.input,
        args.formats,
        args.outdir,
        batch_size=args.batch_size,
        sr=args.sr,
        device=args.device,
        compute_type=args.compute_type,
        lid_margin=args.lid_margin,
        lid_temp=args.lid_temp,
        role_map=args.role_map,
    )
