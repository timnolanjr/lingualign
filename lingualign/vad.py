# lingualign/vad.py

"""
Voice Activity Detection (VAD) utilities for Lingualign.

This module finds “speech turns” in any audio file by:
  1) Loading & normalizing via `io.load_and_normalize`
  2) Splitting into short frames (e.g. 30 ms)
  3) Running WebRTC-VAD on each frame
  4) Tracking voiced/unvoiced transitions to emit (start, end) times
  5) Optionally padding each segment to avoid clipping phonemes

Dependencies:
  - webrtcvad (pip install webrtcvad)
  - numpy
  - soundfile
  - your existing lingualign.io module
"""

import webrtcvad
import numpy as np
from typing import List, Tuple
from pathlib import Path

from .io import load_and_normalize

# Instantiate a global VAD object; aggressiveness=3 is the most aggressive
_VAD = webrtcvad.Vad(3)

def get_speech_turns(
    path: str,
    frame_ms: float = 30.0,
    padding_s: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Detect contiguous speech segments in an audio file.

    Args:
        path (str):
            Path to the input audio/video file. Any format ffmpeg supports.
        frame_ms (float, optional):
            Frame length for VAD, in milliseconds. Must be one of
            [10, 20, 30]. Smaller frames yield more precise boundaries
            but can be noisier. Defaults to 30.
        padding_s (float, optional):
            Amount of padding (in seconds) to add before and after each
            detected speech turn. Helps avoid chopping off phonemes at
            the edges. Defaults to 0.1 (100 ms).

    Returns:
        List[Tuple[float, float]]:
            A list of (start_time, end_time) pairs in seconds, each
            representing a contiguous speech turn (with padding applied).
            Segments are non-overlapping and in chronological order.

    Raises:
        FileNotFoundError:
            If the file at `path` does not exist.
        ValueError:
            If `frame_ms` is not one of 10, 20, or 30.
        RuntimeError:
            If loading or decoding the file fails in `io.load_and_normalize`.

    Example:
        >>> from lingualign.vad import get_speech_turns
        >>> turns = get_speech_turns("podcast.mp3", frame_ms=20, padding_s=0.2)
        >>> for start,end in turns:
        ...     print(f"Speech from {start:.2f}s to {end:.2f}s")
    """
    # Validate inputs
    if frame_ms not in (10.0, 20.0, 30.0):
        raise ValueError("frame_ms must be one of 10.0, 20.0, or 30.0")

    # Load & normalize audio to mono 16 kHz float32
    audio, sr = load_and_normalize(path, sr=16_000, mono=True)

    # Convert padding to frames
    frame_len = int(sr * (frame_ms / 1000.0))
    pad_frames = int(padding_s * sr)

    segments: List[Tuple[float, float]] = []
    voiced = False
    start_frame = 0

    # Process each frame
    for i in range(0, len(audio), frame_len):
        frame = audio[i : i + frame_len]
        # If the last frame is shorter than frame_len, zero-pad it:
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)), mode="constant")

        # WebRTC VAD expects 16-bit PCM little-endian bytes
        pcm = (frame * 32767).astype(np.int16).tobytes()
        is_speech = _VAD.is_speech(pcm, sr)

        t = i / sr
        if is_speech and not voiced:
            # Voice just started
            voiced = True
            # subtract padding, clamp at 0
            start_frame = max(0, i - pad_frames)
        elif not is_speech and voiced:
            # Voice just ended
            voiced = False
            end_frame = min(len(audio), i + pad_frames)
            segments.append((start_frame / sr, end_frame / sr))

    # If file ends while still voiced, close out the last segment
    if voiced:
        segments.append((start_frame / sr, len(audio) / sr))

    return segments
