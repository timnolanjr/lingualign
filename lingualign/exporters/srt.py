"""
Produce an .srt subtitle file.  Spanish tokens get <b>…</b>.
"""

from typing import List, Dict
import datetime


def _sec_to_srt_timestamp(t: float) -> str:
    # Convert seconds float to "HH:MM:SS,mmm"
    td = datetime.timedelta(seconds=t)
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def to_srt(segments: List[Dict], out_path: str) -> None:
    """
    segments: output of assign_word_speakers+annotate_segments_language
    """
    entries = []
    idx = 1
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        # wrap words
        words = []
        for w in seg["words"]:
            tok = w["word"]
            if w.get("lang") == "es":
                tok = f"<b>{tok}</b>"
            words.append(tok)
        text = " ".join(words)
        entries.append(
            "\n".join([
                str(idx),
                f"{_sec_to_srt_timestamp(start)} --> {_sec_to_srt_timestamp(end)}",
                text,
                ""
            ])
        )
        idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
