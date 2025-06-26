# generate_lexicons.py

"""
Generate English and Spanish lexicon files for Lingualign Word‐LID.
Requires: pip install wordfreq
"""

from wordfreq import top_n_list

def build_lexicon(lang_code: str, n_top: int = 50_000, out_path: str = None):
    """
    Fetch top-n frequent words for a language and write to a file.
    
    Args:
        lang_code: BCP-47 code, e.g. "en" or "es"
        n_top: number of top words to include
        out_path: filename to write (default is "<lang_code>.txt")
    """
    out_path = out_path or f"{lang_code}.txt"
    words = top_n_list(lang_code, n_top)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w.lower() + "\n")
    print(f"Wrote {len(words)} words to {out_path}")

if __name__ == "__main__":
    # Generate English and Spanish lexica
    build_lexicon("en", n_top=50000, out_path="english.txt")
    build_lexicon("es", n_top=50000, out_path="spanish.txt")
