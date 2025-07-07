#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import whisperx
from lingualign.io import load_and_normalize
from lingualign.lid import init_audio_lid, annotate_segments_language

from lingualign.exporters.plain import to_plain
from lingualign.exporters.markdown import to_markdown
from lingualign.exporters.srt import to_srt
from lingualign.exporters.tex import to_tex

def map_speakers_to_roles(segments, roles_json_path):
    with open(roles_json_path, encoding="utf-8") as f:
        role_map = json.load(f)
    for seg in segments:
        raw = seg.get("speaker", "")
        human = role_map.get(raw, raw)
        seg["speaker"] = human
        for w in seg["words"]:
            w["speaker"] = human

def run_one(
    audio_path: str,
    output_root: str,
    formats: list[str],
    batch_size: int,
    sb_device: str,
    pipeline_device: str,
    tex_compile: bool,
    margin: float,
):
    stem = Path(audio_path).stem
    outdir = Path(output_root) / stem
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load & normalize audio
    audio, sr = load_and_normalize(audio_path, sr=16000, mono=True)

    # 2) ASR
    asr = whisperx.load_model("medium", device=pipeline_device, compute_type="float32")
    print(f"▶ Transcribing {stem}…")
    result = asr.transcribe(audio, batch_size=batch_size)
    print(f"   → {len(result['segments'])} ASR segments")

    # 3) align
    print("▶ Aligning word‐level timestamps…")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=pipeline_device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        pipeline_device,
        return_char_alignments=False,
    )["segments"]
    print(f"   → {len(aligned)} aligned segments")

    # 4) diarize
    print("▶ Speaker diarization…")
    diarizer = whisperx.diarize.DiarizationPipeline(device=pipeline_device)
    turns = diarizer(audio)
    dia = whisperx.assign_word_speakers(turns, {"segments": aligned})
    segments = dia["segments"]

    # map SPEAKER_00 → Student, etc.
    roles_json = Path(__file__).parent.parent / "roles.json"
    map_speakers_to_roles(segments, roles_json)
    print(f"   → mapped speakers → human roles")

    # 5) language‐ID
    print("▶ Running per-word LID…")
    sb = init_audio_lid(device=sb_device)
    annotated = annotate_segments_language(
        segments,
        audio,
        sr=sr,
        en_lex_path="lexicons/english.txt",
        es_lex_path="lexicons/spanish.txt",
        model=sb,
        margin=margin,
    )
    print("   → LID complete")

    # 6) export
    if "txt" in formats or "plain" in formats:
        to_plain(annotated, audio_path, outdir)
    if "md" in formats or "markdown" in formats:
        to_markdown(annotated, audio_path, outdir)
    if "srt" in formats:
        to_srt(annotated, audio_path, outdir)
    if "tex" in formats:
        to_tex(annotated, audio_path, outdir, compile_pdf=tex_compile)

def main():
    parser = argparse.ArgumentParser(
        description="Lingualign CLI — ASR → Align → Diarize → LID → export"
    )
    parser.add_argument(
        "-f", "--formats",
        nargs="+",
        choices=["markdown", "md", "plain", "srt", "tex", "txt"],
        required=True,
        help="Which output formats to produce"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Root folder under which each track gets its own subfolder"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="ASR batch size"
    )
    parser.add_argument(
        "--device-asr", default="cpu",
        help="Device for WhisperX"
    )
    parser.add_argument(
        "--device-lid", default="cpu",
        help="Device for SpeechBrain LID"
    )
    parser.add_argument(
        "--no-pdf", action="store_false", dest="tex_compile",
        help="Skip running pdflatex in tex"
    )
    parser.add_argument(
        "--margin", type=float, default=0.75,
        help="Text‐based LID cutoff; below this, fall back to audio"
    )
    parser.add_argument(
        "audio_files", nargs="+",
        help="One or more audio files to process"
    )

    args = parser.parse_args()
    for f in args.audio_files:
        run_one(
            f,
            args.output,
            args.formats,
            args.batch_size,
            args.device_lid,
            args.device_asr,
            args.tex_compile,
            args.margin,
        )

if __name__ == "__main__":
    main()
