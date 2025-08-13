# lingualign/lid.py
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import difflib
import numpy as np

try:
    from speechbrain.inference.classifiers import EncoderClassifier
    HAS_SPEECHBRAIN = True
except Exception:
    HAS_SPEECHBRAIN = False

try:
    from wordfreq import zipf_frequency
    HAS_WORDFREQ = True
except Exception:
    HAS_WORDFREQ = False

from .lexicon import load_lexica
from .lang_registry import normalize_lang

@dataclass
class LIDConfig:
    languages: Optional[List[str]] = None
    lexicon_dir: str = 'lexicons'
    margin: float = 0.75
    fuzzy_cutoff: float = 0.85
    pad: float = 0.10
    sr: int = 16000
    debug: bool = False

def _zipf_probs(token: str, langs: List[str]) -> Dict[str, float]:
    if not HAS_WORDFREQ or not langs:
        return {}
    base = {lg: (10.0 ** zipf_frequency(token, lg)) for lg in langs}
    tot = sum(base.values()) or 1.0
    return {lg: v / tot for lg, v in base.items()}

def _extract_audio_chunk(audio: np.ndarray, sr: int, start_s: float, end_s: float, pad: float) -> np.ndarray:
    ns = audio.shape[-1]
    a = max(0, int(sr * (start_s - pad)))
    b = min(ns, int(sr * (end_s + pad)))
    return audio[..., a:b]

def annotate_segments_language(segments: List[Dict], audio: np.ndarray | str | None, model: Optional['EncoderClassifier'], config: LIDConfig) -> List[Dict]:
    langs = sorted({normalize_lang(x) for x in (config.languages or [])})
    tries = load_lexica(langs, config.lexicon_dir)

    for seg in segments:
        words = seg.get('words') or []
        for w in words:
            tok = (w.get('word') or '').strip().lower()
            if not tok:
                continue

            exact_hits = [lg for lg, lex in tries.items() if lex.contains(tok)]
            if len(exact_hits) == 1:
                w['lang'], w['lang_confidence'] = exact_hits[0], 1.0
                if config.debug: print(f"[LEX] {tok!r} -> {w['lang']}")
                continue

            other_vocab = []
            for lg, lex in tries.items():
                other_vocab.extend(lex.words)
            if other_vocab:
                fuzzy = difflib.get_close_matches(tok, other_vocab, n=1, cutoff=config.fuzzy_cutoff)
                if fuzzy:
                    for lg, lex in tries.items():
                        if fuzzy[0] in lex.words:
                            w['lang'], w['lang_confidence'] = lg, 0.9
                            if config.debug: print(f"[FUZZY] {tok!r} -> {lg} (fuzzy:{fuzzy[0]!r})")
                            break
                    if 'lang' in w:
                        continue

            zp = _zipf_probs(tok, langs)
            if zp:
                lg_best, p_best = max(zp.items(), key=lambda x: x[1])
                if p_best >= config.margin:
                    w['lang'], w['lang_confidence'] = lg_best, float(p_best)
                    if config.debug: print(f"[ZIPF] {tok!r} -> {lg_best} ({p_best:.2f})")
                    continue

            if HAS_SPEECHBRAIN and model is not None and isinstance(audio, np.ndarray):
                wstart = float(w.get('start', seg.get('start', 0.0)))
                wend   = float(w.get('end', seg.get('end', wstart + 0.2)))
                chunk = _extract_audio_chunk(audio, config.sr, wstart, wend, config.pad)
                if chunk.size > 0:
                    try:
                        out = model.classify_batch(chunk[np.newaxis, :])
                        probs = getattr(out, 'probabilities', None) or getattr(out, 'p', None)
                        if probs is not None:
                            texts = out[2] if isinstance(out, tuple) and len(out) >= 3 else []
                            if not texts and hasattr(out, 'text_lab'):
                                texts = out.text_lab
                            temp_map: Dict[str, float] = {}
                            for i, lab in enumerate(texts):
                                code = lab.split(':', 1)[0].strip().lower()
                                temp_map[code] = float(probs[0][i])
                            cand = {k: v for k, v in temp_map.items() if not langs or k in langs}
                            if not cand:
                                cand = temp_map
                            best = max(cand.items(), key=lambda x: x[1])
                            w['lang'], w['lang_confidence'] = best[0], float(best[1])
                            if config.debug: print(f"[SB] {tok!r} -> {w['lang']} ({w['lang_confidence']:.2f})")
                            continue
                    except Exception as e:
                        if config.debug: print(f"[SB-ERR] {e}")

            if langs:
                w['lang'], w['lang_confidence'] = langs[0], 0.1
            else:
                w['lang'], w['lang_confidence'] = 'und', 0.0

    return segments
