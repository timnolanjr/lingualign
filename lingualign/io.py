# lingualign/io.py

import subprocess
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import Tuple

def load_and_normalize(
    path: str,
    sr: int = 16_000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Decode any audio/video file into a mono numpy array at the target sample rate.

    Uses ffmpeg under the hood to handle formats like MP4, MP3, WAV, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    # Build the ffmpeg command:
    cmd = [
        "ffmpeg",
        "-y",               # overwrite output if needed
        "-i", str(path),    # input file
        "-vn",              # drop any video stream
        "-ac", "1" if mono else "2",  # audio channels
        "-ar", str(sr),     # sampling rate
        "-f", "wav",        # output format
        "pipe:1",           # send to stdout
    ]

    # Run ffmpeg, capture raw WAV bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.decode('utf-8')}")

    # Read WAV from in-memory buffer
    data, file_sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    # Ensure correct sample rate
    if file_sr != sr:
        raise RuntimeError(f"Unexpected sample rate: {file_sr} != {sr}")

    return data, sr
