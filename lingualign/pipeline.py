#!/usr/bin/env python3
import argparse
import json
import time
import datetime
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
        raw = seg["speaker"]
        human = role_map.get(raw, raw)
        seg["speaker"] = human
        for w in seg["words"]:
            w["speaker"] = human


def run_one(
    audio_path: str,
    output_root: str,
    formats: list[str],
    batch_size: int,
    device_asr: str,
    device_lid: str,
    compile_pdf: bool,
    margin: float,
):
    t0 = time.perf_counter()
    stem   = Path(audio_path).stem
    outdir = Path(output_root) / stem
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load & normalize audio (ffmpeg handles video too)
    audio, sr = load_and_normalize(audio_path, sr=16000, mono=True)

    # 2) ASR w/ CUDA/FP16 → CPU/FP32 fallback
    compute_type = "float16"
    print(f"▶ Loading WhisperX ASR model 'medium' on {device_asr}/{compute_type}…")
    try:
        asr = whisperx.load_model("medium", device=device_asr, compute_type=compute_type)
    except ValueError as e:
        print(f"⚠️  {e} — falling back to CPU/float32")
        device_asr   = "cpu"
        compute_type = "float32"
        asr = whisperx.load_model("medium", device=device_asr, compute_type=compute_type)

    print(f"▶ Transcribing {stem}…")
    result = asr.transcribe(audio, batch_size=batch_size)
    print(f"   → {len(result['segments'])} ASR segments")

    # 3) Align word‐level timestamps
    print("▶ Aligning word‐level timestamps…")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device_asr
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device_asr,
        return_char_alignments=False,
    )["segments"]
    print(f"   → {len(aligned)} aligned segments")

    # 4) Speaker diarization
    print("▶ Speaker diarization…")
    diarizer = whisperx.diarize.DiarizationPipeline(device=device_asr)
    turns = diarizer(audio)
    dia   = whisperx.assign_word_speakers(turns, {"segments": aligned})
    segments = dia["segments"]

    # 4a) Re‐index raw speaker IDs into SPEAKER_00, SPEAKER_01, … in order seen
    raw2canon = {}
    for seg in segments:
        raw = seg.get("speaker", "")
        if raw not in raw2canon:
            raw2canon[raw] = f"SPEAKER_{len(raw2canon):02d}"
        canon = raw2canon[raw]
        seg["speaker"] = canon
        for w in seg["words"]:
            # ensure every word carries the segment’s speaker
            w["speaker"] = canon

    # 4b) Map those SPEAKER_** into human roles via roles.json
    roles_json = Path(__file__).parent.parent / "roles.json"
    map_speakers_to_roles(segments, roles_json)
    print("   → mapped speakers → human roles")

    # 5) per-word Language ID
    print("▶ Initializing SpeechBrain LID model on", device_lid)
    sb = init_audio_lid(device=device_lid)
    print("▶ Running per-word LID…")
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

    # 6) Export into all requested formats
    if "txt" in formats or "plain" in formats:
        to_plain(annotated, audio_path, outdir)
    if "md" in formats or "markdown" in formats:
        to_markdown(annotated, audio_path, outdir)
    if "srt" in formats:
        to_srt(annotated, audio_path, outdir)
    if "tex" in formats:
        to_tex(annotated, audio_path, outdir, compile_pdf=compile_pdf)

    # 7) report elapsed time
    elapsed = time.perf_counter() - t0
    elapsed_str = str(datetime.timedelta(seconds=elapsed)).split(".")[0]
    print(f"▶ Finished `{Path(audio_path).name}` in {elapsed_str}\n")


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
        "--device-asr", dest="device_asr", default="cpu",
        help="Device for WhisperX (e.g. cuda or cpu)"
    )
    parser.add_argument(
        "--device-lid", dest="device_lid", default="cpu",
        help="Device for SpeechBrain LID"
    )
    parser.add_argument(
        "--no-pdf", action="store_false", dest="compile_pdf",
        help="Skip running pdflatex in tex"
    )
    parser.add_argument(
        "--margin", type=float, default=0.75,
        help="Text‐based LID cutoff; below this, fall back to audio"
    )
    parser.add_argument(
        "audio_paths", nargs="+",
        help="One or more audio files or directories to process"
    )

    args = parser.parse_args()

    # Expand any directories into their audio files
    all_inputs = []
    for p in args.audio_paths:
        p = Path(p)
        if p.is_dir():
            for ext in ("mp3", "m4a", "wav", "mp4", "mov"):
                all_inputs.extend(sorted(p.glob(f"*.{ext}")))
        else:
            all_inputs.append(p)

    # Process each in turn
    for audio_file in all_inputs:
        run_one(
            str(audio_file),
            args.output,
            args.formats,
            args.batch_size,
            args.device_asr,
            args.device_lid,
            args.compile_pdf,
            args.margin,
        )


if __name__ == "__main__":
    main()
