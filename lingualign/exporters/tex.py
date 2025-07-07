# lingualign/exporters/tex.py

import subprocess
from pathlib import Path
from typing import List, Dict, Optional

def sanitize_latex(s: str) -> str:
    """
    Escape LaTeX special characters and normalize quotes.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # normalize quotes
    s = s.replace("“", "``").replace("”", "''").replace('"', "''")
    return s

def to_tex(
    segments: List[Dict],
    audio_path: str,
    output_dir: str,
    compile_pdf: bool = True,
    title: Optional[str] = None,
) -> Path:
    """
    Render a LaTeX screenplay (.tex + optionally .pdf).
    Spanish words become \\textbf{\\textit{…}}.
    """

    output_dir = Path(output_dir)
    stem       = Path(audio_path).stem

    # If they already passed .../results/Track_02_90s as output_dir,
    # don't do results/Track_02_90s/Track_02_90s again.
    if output_dir.name == stem:
        track_dir = output_dir
    else:
        track_dir = output_dir / stem

    track_dir.mkdir(parents=True, exist_ok=True)

    doc_title    = sanitize_latex(title or stem)
    parent_title = sanitize_latex(Path(audio_path).parent.name)

    lines: List[str] = [
        r"\documentclass{screenplay}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        rf"\begin{{center}}{{\LARGE\bfseries {parent_title}}}\end{{center}}",
        "",
        rf"\begin{{center}}{{\Large\bfseries {doc_title}}}\end{{center}}",
        ""
    ]

    for seg in segments:
        role = sanitize_latex(seg.get("speaker", "UNK"))
        lines.append(rf"\begin{{dialogue}}{{{role}}}")
        toks: List[str] = []
        for w in seg["words"]:
            tok = sanitize_latex(w["word"])
            if w.get("lang") == "es":
                tok = r"\textbf{\textit{" + tok + "}}"
            toks.append(tok)
        lines.append(" ".join(toks))
        lines.append(r"\end{dialogue}")
        lines.append("")

    lines.append(r"\end{document}")

    tex_path = track_dir / f"{stem}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"▶ Writing LaTeX  → {tex_path}")

    if compile_pdf:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=batchmode", tex_path.name],
                cwd=str(track_dir),
                check=True,
            )
        pdf_path = track_dir / f"{stem}.pdf"
        print(f"✔ PDF generated → {pdf_path}")
        return pdf_path

    return tex_path
