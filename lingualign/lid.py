# lingualign/lid.py
"""
Sliding-window language identification (LID) for Lingualign.
Segments bilingual (English/Spanish) audio into labeled spans.
"""
import numpy as np
import torch
import whisper
from typing import List, Tuple

# Initialize Whisper model for language identification
device = "cuda" if torch.cuda.is_available() else "cpu"
_model = whisper.load_model("base").to(device)


def slide_windows(
    audio: np.ndarray,
    sr: int,
    window_size: float,
    hop_size: float
) -> List[np.ndarray]:
    """
    Slice audio into overlapping windows.

    Args:
        audio: 1-D float32 array of audio samples.
        sr: Sample rate (e.g., 16000).
        window_size: Window length in seconds.
        hop_size: Hop length in seconds.

    Returns:
        List of audio windows as numpy arrays.
    """
    ws = int(window_size * sr)
    hs = int(hop_size * sr)
    windows: List[np.ndarray] = []
    for start in range(0, len(audio) - ws + 1, hs):
        windows.append(audio[start : start + ws])
    return windows


def detect_language(
    window: np.ndarray,
    sr: int
) -> Tuple[str, float]:
    """
    Detect the language of a short audio window, but restrict to EN vs. ES.

    Args:
        window: Audio window numpy array (float32).
        sr:    Sample rate (must be 16 kHz for Whisper).

    Returns:
        (language_code, confidence_score) where language_code ∈ {"en","es","un"}
    """
    # 1) Convert to Whisper’s expected input
    audio_tensor = torch.from_numpy(window).to(device)
    audio_tensor = whisper.pad_or_trim(audio_tensor)

    # 2) Compute log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_tensor).to(device)

    # 3) Get the full whisper LID probability map
    #    detect_language returns (detected, probs) but we only need probs
    _, lang_probs = _model.detect_language(mel)

    # 4) Pull out just English and Spanish
    en_prob = float(lang_probs.get("en", 0.0))
    es_prob = float(lang_probs.get("es", 0.0))

    # 5) Pick the winner (or mark 'un' if both are zero)
    if en_prob == 0 and es_prob == 0:
        return "un", 0.0
    if en_prob >= es_prob:
        return "en", en_prob
    else:
        return "es", es_prob




def label_windows(
    windows: List[np.ndarray],
    sr: int,
    threshold: float,
    window_size: float,
    hop_size: float
) -> List[Tuple[float, float, str, float]]:
    """
    Label each window with a language code and confidence.

    Args:
        windows: List of audio windows.
        sr: Sample rate (unused, for API symmetry).
        threshold: Minimum confidence to accept label.
        window_size: Window length in seconds.
        hop_size: Hop length in seconds.

    Returns:
        List of tuples (t_start, t_end, lang, confidence).
    """
    spans: List[Tuple[float, float, str, float]] = []
    for idx, w in enumerate(windows):
        lang, conf = detect_language(w, sr)
        # Map low-confidence or out-of-domain to 'un' (uncertain)
        if conf < threshold or lang not in ("en", "es"):
            lang = "un"
        t_start = idx * hop_size
        t_end = t_start + window_size
        spans.append((t_start, t_end, lang, conf))
    return spans


def merge_labels(
    spans: List[Tuple[float, float, str, float]]
) -> List[Tuple[float, float, str]]:
    """
    Merge only adjacent spans with the identical language label.
    Leave all 'un' segments (and any other brief flips) intact.

    Args:
        spans: List of (t_start, t_end, lang, conf).

    Returns:
        List of merged segments as (t_start, t_end, lang).
    """
    if not spans:
        return []

    merged: List[Tuple[float, float, str]] = []
    cur_start, cur_end, cur_lang, _ = spans[0]

    for start, end, lang, _ in spans[1:]:
        if lang == cur_lang:
            # same language → extend the current segment
            cur_end = end
        else:
            # different language → emit the current, start a new one
            merged.append((cur_start, cur_end, cur_lang))
            cur_start, cur_end, cur_lang = start, end, lang

    # emit the final segment
    merged.append((cur_start, cur_end, cur_lang))
    return merged



def segment_audio(
    audio: np.ndarray,
    sr: int,
    window_size: float,
    hop_size: float,
    threshold: float,
    min_duration: float
) -> List[Tuple[np.ndarray, float, float, str]]:
    """
    Full LID pipeline: windowing, labeling, and merging.

    Args:
        audio: Full audio array.
        sr: Sample rate.
        window_size: Window length (s).
        hop_size: Hop length (s).
        threshold: Confidence threshold for labels.
        min_duration: Minimum duration (s) to keep segments.

    Returns:
        List of tuples (segment_audio, t_start, t_end, lang).
    """
    windows = slide_windows(audio, sr, window_size, hop_size)
    spans = label_windows(windows, sr, threshold, window_size, hop_size)
    merged = merge_labels(spans)

    segments: List[Tuple[np.ndarray, float, float, str]] = []
    for t_start, t_end, lang in merged:
        start_idx = int(t_start * sr)
        end_idx = int(t_end * sr)
        segment = audio[start_idx:end_idx]
        segments.append((segment, t_start, t_end, lang))
    return segments
