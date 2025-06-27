"""
Produce a Markdown‐flavored transcript, bolding Spanish tokens.
"""

from typing import List, Dict

def to_markdown(
    segments: List[Dict],
    out_path: str,
    bold_tag: str = "**"
) -> None:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNK")
        tokens = []
        for w in seg["words"]:
            tok = w["word"]
            if w.get("lang") == "es":
                tok = f"{bold_tag}{tok}{bold_tag}"
            tokens.append(tok)
        lines.append(f"**{speaker}**: {' '.join(tokens)}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
