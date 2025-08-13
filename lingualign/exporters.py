# lingualign/exporters.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional

import shutil
import subprocess

def to_plain(segments: List[Dict], audio_path: str, output_dir: Path, highlight_langs: Optional[list[str]] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (Path(audio_path).stem + '.txt')
    hl = set(highlight_langs or [])
    with out.open('w', encoding='utf-8') as f:
        for seg in segments:
            line = []
            for w in seg.get('words', []):
                tok = str(w.get('word', ''))
                if hl and w.get('lang') in hl:
                    tok = f"***{tok}***"
                line.append(tok)
            f.write(' '.join(line).strip() + '\n')
    return out

def to_markdown(segments: List[Dict], audio_path: str, output_dir: Path, highlight_langs: Optional[list[str]] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (Path(audio_path).stem + '.md')
    hl = set(highlight_langs or [])
    with out.open('w', encoding='utf-8') as f:
        for seg in segments:
            line = []
            for w in seg.get('words', []):
                tok = str(w.get('word', ''))
                if hl and w.get('lang') in hl:
                    tok = f"***{tok}***"
                line.append(tok)
            f.write('- ' + ' '.join(line).strip() + '\n')
    return out

def to_srt(segments: List[Dict], audio_path: str, output_dir: Path, highlight_langs: Optional[list[str]] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (Path(audio_path).stem + '.srt')
    hl = set(highlight_langs or [])
    def fmt(t: float) -> str:
        h = int(t // 3600); t %= 3600
        m = int(t // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
    idx = 1
    with out.open('w', encoding='utf-8') as f:
        for seg in segments:
            start = fmt(float(seg.get('start', 0.0)))
            end   = fmt(float(seg.get('end', 0.0)))
            f.write(f"{idx}\n{start} --> {end}\n")
            line = []
            for w in seg.get('words', []):
                tok = str(w.get('word', ''))
                if hl and w.get('lang') in hl:
                    tok = f"<i><b>{tok}</b></i>"
                line.append(tok)
            f.write((' '.join(line).strip() or '').strip() + '\n\n')
            idx += 1
    return out

def _latex_escape(text: str) -> str:
    """
    Escape a minimal but practical set of LaTeX special chars.
    Keeps it readable while avoiding compilation issues.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)

def to_tex(
    segments: List[Dict],
    audio_path: str,
    output_dir: Path,
    compile_pdf: bool = False,
    title: Optional[str] = None,
    highlight_langs: Optional[list[str]] = None,
) -> Path:
    """
    Render a LaTeX transcript with optional per-language highlighting.
    Highlighted words are wrapped in \textbf{\textit{...}}.

    Returns the path to the .tex file. If compile_pdf=True and pdflatex is
    available, also produces a .pdf next to it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / (Path(audio_path).stem + ".tex")
    hl = set(highlight_langs or [])

    with tex_path.open("w", encoding="utf-8") as f:
        f.write(
            r"\documentclass[11pt]{article}" "\n"
            r"\usepackage[utf8]{inputenc}" "\n"
            r"\usepackage[T1]{fontenc}" "\n"
            r"\usepackage{geometry}" "\n"
            r"\geometry{margin=1in}" "\n"
            r"\usepackage{microtype}" "\n"
            r"\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue]{hyperref}" "\n"
            r"\begin{document}" "\n"
        )
        if title:
            f.write(r"\section*{" + _latex_escape(title) + "}\n")

        # one line per segment
        for seg in segments:
            line_tokens = []
            for w in seg.get("words", []):
                tok = str(w.get("word", "")).strip()
                if not tok:
                    continue
                tok = _latex_escape(tok)
                if hl and w.get("lang") in hl:
                    tok = r"\textbf{\textit{" + tok + "}}"
                line_tokens.append(tok)

            line = " ".join(line_tokens).strip()
            if not line:
                line = r"\mbox{}"  # safe empty line
            f.write(line + r" \\ " + "\n")

        f.write(r"\end{document}" "\n")

    if compile_pdf:
        exe = shutil.which("pdflatex")
        if exe is not None:
            # Run pdflatex in the output directory
            cmd = [exe, "-interaction=nonstopmode", tex_path.name]
            try:
                subprocess.run(cmd, cwd=str(output_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                # If compilation fails, just return the .tex; caller can inspect manually
                pass
        # else: silently skip compilation

    return tex_path