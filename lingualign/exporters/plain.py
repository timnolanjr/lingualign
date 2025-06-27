"""
Produce a plain‐text transcript:

Speaker: word word word…
"""

from typing import List, Dict

def to_plain(segments: List[Dict], out_path: str) -> None:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNK")
        words = [w["word"] for w in seg["words"]]
        lines.append(f"{speaker}: {' '.join(words)}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
