# lingualign/io.py

"""
I/O utilities for Lingualign: decoding any audio/video file
into a normalized NumPy array ready for downstream processing.

This module depends on:
  - ffmpeg (installed and on your PATH)
  - soundfile (PySoundFile)
  - numpy
"""

import subprocess
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_and_normalize(
    path: str,
    sr: int = 16_000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Decode an audio or video file into a NumPy array of float32 samples.

    This function invokes ffmpeg under the hood to handle virtually any
    input format (MP3, M4A, MP4, WAV, etc.), resamples to `sr`, and
    outputs either mono or stereo according to `mono`.

    Args:
        path (str):
            Path to the input file. Can be any format ffmpeg supports.
        sr (int, optional):
            Desired sampling rate in Hz for the output array.
            Defaults to 16000.
        mono (bool, optional):
            If True, downmix to a single channel. If False, preserves
            two channels. Defaults to True.

    Returns:
        Tuple[np.ndarray, int]:
            - data: 1-D (mono) or 2-D (stereo) NumPy array of dtype float32,
              with values in [-1.0, 1.0].
            - sr: The sampling rate of the returned data (will equal the
              requested `sr` on success).

    Raises:
        FileNotFoundError:
            If `path` does not exist.
        RuntimeError:
            If ffmpeg fails (non-zero exit code), or if the decoded
            sample rate does not match the requested `sr`.

    Example:
        >>> audio, fs = load_and_normalize("podcast.mp3", sr=16000)
        >>> audio.shape
        (for mono)    # e.g. (320000,)
        (for stereo)  # e.g. (2, 320000)

        >>> fs
        16000
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    # Construct ffmpeg command to decode and resample
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output if needed
        "-i", str(path),      # input file
        "-vn",                # drop any video streams
        "-ac", "1" if mono else "2",  # 1=mono, 2=stereo
        "-ar", str(sr),       # output sampling rate
        "-f", "wav",          # output format
        "pipe:1",             # write to stdout
    ]

    # Run ffmpeg and capture stdout/stderr
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed:\n{err}")

    # Read the WAV bytes from ffmpeg via an in-memory buffer
    data, file_sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")

    # Sanity check: ensure we got the expected sample rate
    if file_sr != sr:
        raise RuntimeError(f"Unexpected sample rate: {file_sr} != {sr}")

    return data, sr
