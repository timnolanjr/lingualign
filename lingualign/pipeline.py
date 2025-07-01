#!/usr/bin/env python3
"""
lingualign/pipeline.py

Top‐level driver for transcription + segmentation + LID + export.
"""

import os
import argparse
import warnings
from typing import List, Dict, Optional
from collections import defaultdict

import whisperx

from lingualign.io import load_and_normalize
from lingualign.lid import init_audio_lid, annotate_segments_language
from lingualign.exporters import to_plain, to_markdown, to_srt, to_tex

warnings.filterwarnings("ignore", message=r"You are using `torch.load`")
warnings.filterwarnings("ignore", message=r"No language specified.*")
warnings.filterwarnings("ignore", message=r"audio is shorter than 30s.*")


def transcribe_and_align(
    audio,
    sr: int,
    model: "whisperx.WhisperModel",
    batch_size: int,
    device: str,
) -> List[Dict]:
    """Run WhisperX ASR → word‐level alignment."""
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
    audio,
    device: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict]:
    """
    Run speaker diarization, forcing min/max speakers if provided.
    """
    diarizer = whisperx.diarize.DiarizationPipeline(device=device)
    turns = diarizer(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    dia = whisperx.assign_word_speakers(turns, {"segments": aligned})
    segments = dia["segments"]
    # fill missing word‐level speaker tags from segment speaker
    for seg in segments:
        seg_spk = seg.get("speaker", "UNK")
        for w in seg.get("words", []):
            if not w.get("speaker"):
                w["speaker"] = seg_spk
    return segments


def run_pipeline(
    input_path: str,
    formats: List[str],
    output_dir: str,
    batch_size: int,
    sr: int,
    device: str,
    compute_type: str,
    lid_margin: float,
    lid_temp: float,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"▶ Decoding audio ({input_path!r}) → {sr} Hz mono")
    audio, _ = load_and_normalize(input_path, sr=sr, mono=True)

    print("▶ Loading WhisperX ASR model…")
    asr_model = whisperx.load_model(
        "medium", device=device, compute_type=compute_type
    )

    print("▶ Transcribing & aligning…")
    aligned = transcribe_and_align(audio, sr, asr_model, batch_size, device)

    print("▶ Diarizing…")
    segments = diarize(
        aligned, audio, device, min_speakers, max_speakers
    )

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

    # ─── Dynamic role mapping: whoever speaks first is Teacher, second is Student ───
    speaker_ids = {seg["speaker"] for seg in annotated}
    role_map = None
    if len(speaker_ids) == 2:
        first_times = defaultdict(lambda: float("inf"))
        for seg in annotated:
            spk = seg["speaker"]
            first_times[spk] = min(first_times[spk], seg["start"])
        ordered = sorted(speaker_ids, key=lambda spk: first_times[spk])
        role_map = {
            ordered[0]: "Teacher",
            ordered[1]: "Student"
        }
        print(f"🔑 Dynamic role_map: {role_map}")
    else:
        print(f"ℹ️  Found {len(speaker_ids)} speakers; skipping dynamic mapping.")

    base = os.path.splitext(os.path.basename(input_path))[0]

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
            tex_dir,
            title=base,
            role_map=role_map,
        )

    print("✅ Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Run Lingualign transcript pipeline")
    p.add_argument("input", help="path to audio/video file")
    p.add_argument(
        "-f", "--formats",
        nargs="+",
        choices=["txt", "md", "srt", "tex"],
        default=["txt", "md", "srt", "tex"],
        help="which output formats to generate"
    )
    p.add_argument(
        "-o", "--outdir",
        default="output",
        help="directory to write all outputs into"
    )
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--sr",          type=int,   default=16000)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--compute-type",default="float32")
    p.add_argument(
        "--lid-margin",  type=float, default=0.75,
        help="Zipf‐sigmoid cutoff before invoking audio LID"
    )
    p.add_argument(
        "--lid-temp",    type=float, default=5.0,
        help="temperature for audio‐LID softmax"
    )
    p.add_argument(
        "--min-speakers", type=int, default=None,
        help="(optional) minimum number of speakers for diarization"
    )
    p.add_argument(
        "--max-speakers", type=int, default=None,
        help="(optional) maximum number of speakers for diarization"
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
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
