from pathlib import Path
from typing import List, Dict

def to_plain(
    segments: List[Dict],
    audio_path: str,
    output_dir: Path,
) -> Path:
    """
    Write a simple .txt transcript, grouping by segment.
    Spanish words are wrapped in ***bold+italic*** markers.
    """
    stem = Path(audio_path).stem
    out = output_dir / f"{stem}.txt"

    lines = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        spk = seg.get("speaker", "UNK")
        tokens = []
        for w in seg["words"]:
            tok = w["word"]
            if w.get("lang") == "es":
                tok = f"***{tok}***"
            tokens.append(tok)
        text = " ".join(tokens)
        lines.append(f"{start:.2f}-{end:.2f} | {spk}: {text}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"▶ Writing plain text → {out}")
    return out
