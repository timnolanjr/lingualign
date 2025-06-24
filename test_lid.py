"""
Language Identification (LID) utilities using SpeechBrain ECAPA-TDNN.

This module provides functions to:
  - load a pretrained ECAPA-TDNN language ID model
  - classify individual audio segments as 'en' vs 'es'
  - process VAD segments end-to-end

Dependencies:
  - speechbrain
  - numpy
  - torch
  - your existing `lingualign.vad` and `lingualign.io` modules
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")

import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier

from lingualign.vad import get_speech_turns
from lingualign.io import load_and_normalize

# Load the ECAPA-TDNN VoxLingua107 LID model
LANG_ID = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/lang-id-voxlingua107-ecapa"
)
# Suppress category-count warning
LANG_ID.hparams.label_encoder.ignore_len()

# Retrieve label-to-index mapping
encoder = LANG_ID.hparams.label_encoder
label2ind = getattr(encoder, 'label2ind', None) or getattr(encoder, 'lab2ind', None)
if label2ind is None:
    raise RuntimeError("Cannot find label-to-index mapping on the model's label_encoder")

# Helper to find index for a given ISO prefix ('en' or 'es')
def _find_idx(prefix: str) -> int:
    # Exact match first
    if prefix in label2ind:
        return label2ind[prefix]
    # Fallback: prefix match
    for lang_code, idx in label2ind.items():
        if lang_code.lower().startswith(prefix):
            return idx
    raise KeyError(f"Language prefix '{prefix}' not found in model labels")

EN_IDX = _find_idx('en')
ES_IDX = _find_idx('es')


def classify_segment(audio: np.ndarray, sr: int = 16000, min_duration: float = 0.5):
    """
    Classify a raw audio segment and return ('en' or 'es', confidence).

    Args:
        audio: 1-D NumPy array at sample rate `sr`.
        sr: sample rate (default 16000).
        min_duration: pad segments shorter than this (seconds).

    Returns:
        (iso_code, confidence)
    """
    # Handle empty segments
    if audio.size == 0:
        return 'en', 0.0

    # Convert to torch tensor
    segment = torch.from_numpy(audio).float()
    # Pad if too short
    min_len = int(min_duration * sr)
    if segment.numel() < min_len:
        segment = F.pad(segment, (0, min_len - segment.numel()))
    # Add batch dimension
    segment = segment.unsqueeze(0)

    # Run classification: returns (probs, best_score, best_index, text_labels)
    probs, _, _, _ = LANG_ID.classify_batch(segment)
    probs = probs[0]  # shape: (num_languages,)

    # Compare English vs Spanish
    en_conf = float(probs[EN_IDX])
    es_conf = float(probs[ES_IDX])
    if es_conf > en_conf:
        return 'es', es_conf
    return 'en', en_conf


def label_languages(path: str, frame_ms: float = 30.0, padding_s: float = 0.1):
    """
    Perform VAD on `path`, then classify each speech segment as 'en' or 'es'.

    Returns:
        List of (start_s, end_s, iso_code, confidence).
    """
    turns = get_speech_turns(path, frame_ms=frame_ms, padding_s=padding_s)
    audio, sr = load_and_normalize(path, sr=16000, mono=True)

    results = []
    for start, end in turns:
        start_i, end_i = int(start * sr), int(end * sr)
        segment = audio[start_i:end_i]
        if segment.size == 0:
            continue
        lang, conf = classify_segment(segment, sr)
        results.append((start, end, lang, conf))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VAD + LID pipeline (EN vs ES)")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--frame_ms", type=float, default=30.0,
                        help="VAD frame size in ms (10,20,30)")
    parser.add_argument("--padding_s", type=float, default=0.1,
                        help="Padding seconds before/after speech turns")
    args = parser.parse_args()

    out = label_languages(
        args.audio_path,
        frame_ms=args.frame_ms,
        padding_s=args.padding_s
    )
    if not out:
        print("No speech segments detected.")
    for s, e, lang, conf in out:
        print(f"{s:.2f}s–{e:.2f}s: {lang} ({conf:.2f})")