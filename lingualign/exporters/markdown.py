from pathlib import Path
from typing import List, Dict

def to_markdown(
    segments: List[Dict],
    audio_path: str,
    output_dir: Path,
) -> Path:
    """
    Write a .md transcript, one segment per paragraph.
    Spanish words get ***bold+italic*** emphasis.
    """
    stem = Path(audio_path).stem
    out = output_dir / f"{stem}.md"

    lines = [f"# Transcript: {stem}", ""]
    for seg in segments:
        spk = seg.get("speaker", "UNK")
        tokens = []
        for w in seg["words"]:
            tok = w["word"]
            if w.get("lang") == "es":
                tok = f"***{tok}***"
            tokens.append(tok)
        text = " ".join(tokens)
        lines.append(f"**{spk}**: {text}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"▶ Writing Markdown → {out}")
    return out
