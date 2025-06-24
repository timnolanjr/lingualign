# lingualign/lid.py

"""
Language Identification (LID) utilities using SpeechBrain ECAPA-TDNN.

This module provides functions to:
  - load a pretrained ECAPA-TDNN language ID model
  - classify individual audio segments as English vs. Spanish
  - process VAD segments end-to-end
"""

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from .vad import get_speech_turns
from .io import load_and_normalize

# Load ECAPA-TDNN VoxLingua107 LID model once at import
LANG_ID = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/lang-id-voxlingua107-ecapa",
)

# Suppress the “expect_len” warning if it appears
try:
    LANG_ID.hparams.label_encoder.ignore_len()
except Exception:
    pass


def classify_segment(
    audio: np.ndarray,
    sr: int = 16000,
    min_duration: float = 0.5
) -> (str, float):
    """
    Classify a raw audio segment as English ('en') or Spanish ('es').

    Pads very short segments, runs the ECAPA-TDNN model, and
    maps the predicted language to either 'en' or 'es'.

    Args:
        audio: 1-D numpy array of float32 samples at `sr`
        sr: sampling rate (default: 16000)
        min_duration: minimum segment length in seconds; pad if shorter

    Returns:
        iso_code: 'en' or 'es'
        confidence: float probability between 0 and 1
    """
    # 1) Pad to at least min_duration
    min_len = int(min_duration * sr)
    if len(audio) < min_len:
        padding = min_len - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    # 2) Convert to torch.Tensor; build length tensor
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, time)
    wav_lens = torch.tensor([wav.size(-1)], dtype=torch.long)   # (1,)

    # 3) Move to model's device
    device = LANG_ID.device
    wav, wav_lens = wav.to(device), wav_lens.to(device)

    # 4) Run classification
    _, probs, _, predicted = LANG_ID.classify_batch(wav, wav_lens)
    # predicted[0] is the model's best-label string (e.g. 'Spanish', 'English', etc.)
    best_label = predicted[0].lower()
    confidence = float(probs[0].max())

    # 5) Map any detected Spanish to 'es'; everything else → 'en'
    if 'spanish' in best_label or best_label in ('es', 'spa'):
        return 'es', confidence
    else:
        return 'en', confidence


def label_languages(
    path: str,
    frame_ms: float = 30.0,
    padding_s: float = 0.1
) -> list[tuple[float, float, str, float]]:
    """
    Detect speech turns via VAD and classify each segment as 'en' or 'es'.

    Args:
        path: path to input audio file
        frame_ms: VAD frame size in milliseconds
        padding_s: padding around each segment in seconds

    Returns:
        A list of (start_time, end_time, iso_code, confidence)
    """
    # 1) Find speech segments
    turns = get_speech_turns(path, frame_ms=frame_ms, padding_s=padding_s)

    # 2) Load full audio for slicing
    audio, sr = load_and_normalize(path, sr=16000, mono=True)

    results = []
    for start, end in turns:
        s_idx, e_idx = int(start * sr), int(end * sr)
        segment = audio[s_idx:e_idx]
        iso, conf = classify_segment(segment, sr=sr)
        results.append((start, end, iso, conf))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAD + LID (en vs es)")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument(
        "--frame_ms", type=float, default=30.0, help="VAD frame length (ms)"
    )
    parser.add_argument(
        "--padding_s", type=float, default=0.1, help="Padding around speech turns (s)"
    )
    args = parser.parse_args()

    for start, end, iso, conf in label_languages(
        args.audio_path, frame_ms=args.frame_ms, padding_s=args.padding_s
    ):
        print(f"{start:.2f}s–{end:.2f}s:\t{iso}\t{conf:.2f}")
