from pathlib import Path
from typing import List, Dict

def _sec_to_srt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def to_srt(
    segments: List[Dict],
    audio_path: str,
    output_dir: Path,
) -> Path:
    """
    Write a standard SubRip .srt file.
    Spanish words get <i><b>…</b></i> tags.
    """
    stem = Path(audio_path).stem
    out = output_dir / f"{stem}.srt"

    idx = 1
    lines = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        spk = seg.get("speaker", "UNK")
        tokens = []
        for w in seg["words"]:
            tok = w["word"]
            if w.get("lang") == "es":
                tok = f"<i><b>{tok}</b></i>"
            tokens.append(tok)
        text = " ".join(tokens)

        lines.append(str(idx))
        lines.append(f"{_sec_to_srt(start)} --> {_sec_to_srt(end)}")
        lines.append(f"{spk}: {text}")
        lines.append("")
        idx += 1

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"▶ Writing SRT → {out}")
    return out
