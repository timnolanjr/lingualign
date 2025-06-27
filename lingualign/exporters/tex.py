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
    title: Optional[str] = None,
    role_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Render a LaTeX screenplay (.tex + .pdf) for `segments`.
    Places outputs under output_dir/<stem>/<stem>.tex and .pdf.
    """

    base_out = Path(output_dir) / Path(audio_path).stem
    base_out.mkdir(parents=True, exist_ok=True)

    stem      = Path(audio_path).stem
    doc_title = sanitize_latex(title or stem)

    # 1) Build the preamble + manual title
    lines: List[str] = [
        r"\documentclass{screenplay}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        rf"\begin{{center}}{{\LARGE\bfseries {doc_title}}}\end{{center}}",
        ""
    ]

    # 2) Dialogue blocks
    for seg in segments:
        raw_role = seg.get("speaker", "UNK")
        mapped   = role_map.get(raw_role, raw_role) if role_map else raw_role
        role     = sanitize_latex(mapped)

        lines.append(rf"\begin{{dialogue}}{{{role}}}")
        tokens: List[str] = []
        for w in seg["words"]:
            tok = sanitize_latex(w["word"])
            if w.get("lang") == "es":
                tok = r"\textbf{" + tok + "}"
            tokens.append(tok)
        lines.append(" ".join(tokens))
        lines.append(r"\end{dialogue}")
        lines.append("")

    lines.append(r"\end{document}")

    # 3) Write .tex and compile
    tex_path = base_out / f"{stem}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"▶ Writing LaTeX     → {tex_path}")

    # run pdflatex twice
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=batchmode", tex_path.name],
            cwd=str(base_out),
            check=True,
        )

    pdf_path = base_out / f"{stem}.pdf"
    print(f"✔ PDF generated → {pdf_path}")
    return pdf_path

__all__ = ["to_tex"]
