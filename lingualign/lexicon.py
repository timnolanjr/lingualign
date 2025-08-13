# lingualign/lexicon.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, List

try:
    import marisa_trie
    HAS_MARISA = True
except Exception:
    HAS_MARISA = False

from .lang_registry import normalize_lang, NAME_ALIASES, ALIASES_3_TO_1

@dataclass
class Lexicon:
    lang: str
    words: Set[str]
    trie: Optional['marisa_trie.Trie'] = None

    def contains(self, token: str) -> bool:
        t = token.lower()
        if self.trie is not None:
            return t in self.trie
        return t in self.words

def _candidate_filenames(lang_code: str) -> List[str]:
    cands = [f'{lang_code}.txt']
    rev = {v:k for k,v in ALIASES_3_TO_1.items()}
    if lang_code in rev:
        cands.append(f"{rev[lang_code]}.txt")
    for name, code in NAME_ALIASES.items():
        if code == lang_code:
            cands.append(name.replace(' ','_') + '.txt')
    dedup = []
    seen = set()
    for x in cands:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

def load_lexica(languages: Iterable[str], lexicon_dir: Path | str, min_len: int = 1) -> Dict[str, Lexicon]:
    lexdir = Path(lexicon_dir)
    out: Dict[str, Lexicon] = {}
    for lang in languages:
        code = normalize_lang(lang)
        words: Set[str] = set()
        found_file: Optional[Path] = None
        for fname in _candidate_filenames(code):
            fpath = lexdir / fname
            if fpath.exists():
                found_file = fpath
                break
        if not found_file:
            continue
        with found_file.open('r', encoding='utf-8') as f:
            for ln in f:
                w = ln.strip().lower()
                if len(w) >= min_len:
                    words.add(w)
        trie = None
        if HAS_MARISA:
            try:
                trie = marisa_trie.Trie(list(words))
            except Exception:
                trie = None
        out[code] = Lexicon(lang=code, words=words, trie=trie)
    return out
