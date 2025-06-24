#!/usr/bin/env python3
"""
transcribe.py

VAD → LID → Whisper ASR pipeline using separate English (.en) and multilingual models,
with proper handling of SpeechBrain’s AMP API and Whisper’s FP16 flag.

Usage:
    python transcribe.py /path/to/audio \
        [--en_model_size MODEL] [--es_model_size MODEL]
        [--frame_ms N] [--padding_s N] [--fp16]
"""
import argparse
import tempfile

import soundfile as sf
import torch

# Monkey-patch SpeechBrain’s autocast to use the new torch.amp.custom_fwd API
from speechbrain.utils import autocast as sb_autocast
sb_autocast.custom_fwd = lambda fwd, cast_inputs, device_type='cuda': torch.amp.custom_fwd(
    fwd, cast_inputs, device_type=device_type
)

import whisper

from lingualign.vad import get_speech_turns
from lingualign.io import load_and_normalize
from lingualign.lid import classify_segment


def transcribe_segments(
    audio_path: str,
    model_en,
    model_es,
    frame_ms: float = 30.0,
    padding_s: float = 0.1,
    fp16: bool = False,
):
    """
    Detect speech turns, run LID (en/es), and transcribe each segment
    with the appropriate Whisper model.
    Returns list of (start, end, lang, text).
    """
    audio, sr = load_and_normalize(audio_path, sr=16000, mono=True)
    turns = get_speech_turns(audio_path, frame_ms=frame_ms, padding_s=padding_s)
    results = []

    for start, end in turns:
        s_i, e_i = int(start * sr), int(end * sr)
        segment = audio[s_i:e_i]
        if segment.size == 0:
            continue

        lang, _ = classify_segment(segment, sr)
        model = model_en if lang == "en" else model_es

        # Write to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, segment, sr, subtype="PCM_16")
            res = model.transcribe(tmp.name, language=lang, fp16=fp16)

        text = res.get("text", "").strip()
        results.append((start, end, lang, text))

    return results


def main():
    p = argparse.ArgumentParser(
        description="VAD → LID (en/es) → Whisper, with separate en/.en+ multilingual models"
    )
    p.add_argument("audio_path", help="Input audio path (ffmpeg‐supported)")
    p.add_argument(
        "--en_model_size",
        default="medium.en",
        help="English‐only Whisper model: tiny.en, base.en, small.en, medium.en, large-v1",
    )
    p.add_argument(
        "--es_model_size",
        default="medium",
        help="Multilingual Whisper model (for Spanish): small, medium, large-v1, etc.",
    )
    p.add_argument(
        "--frame_ms",
        type=float,
        default=30.0,
        help="VAD frame size in ms (10, 20, or 30)",
    )
    p.add_argument(
        "--padding_s",
        type=float,
        default=0.1,
        help="Padding around speech turns, in seconds",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 for Whisper (requires GPU support)",
    )
    args = p.parse_args()

    # Load English‐only and multilingual models
    model_en = whisper.load_model(args.en_model_size)
    model_es = whisper.load_model(args.es_model_size)

    # Per‐segment ASR
    segments = transcribe_segments(
        args.audio_path,
        model_en,
        model_es,
        frame_ms=args.frame_ms,
        padding_s=args.padding_s,
        fp16=args.fp16,
    )
    for start, end, lang, text in segments:
        print(f"{start:.2f}s–{end:.2f}s: {lang} \"{text}\"")

    # Full‐file ASR using the multilingual model
    print("\nFull transcript:")
    full = model_es.transcribe(args.audio_path, fp16=args.fp16)
    print(full["text"].strip())


if __name__ == "__main__":
    main()
