"""
Common utilities (e.g. for escaping LaTeX strings).
"""

def sanitize_latex(s: str) -> str:
    replacements = {
        "\\\\": r"\\textbackslash{}",
        "&": r"\\&", "%": r"\\%", "\$": r"\\$", "#": r"\\#",
        "_": r"\\_", "{": r"\\{", "}": r"\\}", "~": r"\\textasciitilde{}",
        "^": r"\\^{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = s.replace("“", "``").replace("”", "''").replace('"', "''")
    return s
