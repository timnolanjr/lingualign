# lingualign/lid.py

import os
import difflib
from typing import List, Dict, Union, Optional

import numpy as np
from tqdm import tqdm
import torch
import whisperx
from wordfreq import zipf_frequency
import marisa_trie
from speechbrain.inference.classifiers import EncoderClassifier
from lingualign.io import load_and_normalize

def load_lexicon_trie(path: str) -> marisa_trie.Trie:
    """
    Read a one-word-per-line file and return a marisa_trie.Trie for fast lookup.
    """
    with open(path, encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return marisa_trie.Trie(words)


def init_audio_lid(
    source: str = "speechbrain/lang-id-voxlingua107-ecapa",
    savedir: str = "pretrained_models/lang-id-voxlingua107-ecapa",
    device: str = "cpu",
) -> EncoderClassifier:
    """
    Load the SpeechBrain VoxLingua107 language‐ID model as an EncoderClassifier,
    and tell its internal label_encoder exactly how many classes it has so it
    won’t warn about unexpected lengths.
    """
    os.makedirs(savedir, exist_ok=True)
    model = EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": device},
    )
    # VoxLingua107 has 107 classes; make sure the encoder knows that
    n_labels = len(model.hparams.label_encoder.ind2lab)
    model.hparams.label_encoder.expect_len(n_labels)
    return model


def annotate_segments_language(
    segments: List[Dict],
    audio: Union[str, np.ndarray],
    sr: int = 16000,
    en_lex_path: str = "lexicons/english.txt",
    es_lex_path: str = "lexicons/spanish.txt",
    model: Optional[EncoderClassifier] = None,
    pad: float = 0.1,
    temp: float = 5.0,
    margin: float = 0.75,
    fuzzy_cutoff: float = 0.8,
    debug: bool = False,
) -> List[Dict]:
    """
    Annotate each word in `segments` with:
      - 'lang': 'en', 'es', or None
      - 'lang_confidence': float [0.0–1.0]

    Steps:
      1) Exact lexicon lookup via marisa-trie (with fuzzy‐in‐other‐list check)
      2) Zipf‐sigmoid tie‐breaker
      3) Audio‐based fallback via SpeechBrain
    """

    # 1) Load lexica
    en_trie = load_lexicon_trie(en_lex_path)
    es_trie = load_lexicon_trie(es_lex_path)

    # 2) Init SpeechBrain LID if needed
    if model is None:
        model = init_audio_lid(device="cpu")

    # 3) Load or accept audio array
    if isinstance(audio, (str, os.PathLike)):
        audio, _ = load_and_normalize(str(audio), sr=sr, mono=True)
    # else: assume already a np.ndarray
    total_dur = len(audio) / sr

    # 4) Annotate each word
    for seg in tqdm(segments, desc="LID segments"):
        for w in seg.get("words", []):
            tok = w["word"].lower().strip(".,?!)('\"")
            in_en = tok in en_trie
            in_es = tok in es_trie

            if debug:
                print(f"[LEX] tok={tok!r}, in_en={in_en}, in_es={in_es}")

            # 4.1) Exact lexicon—but check for a fuzzy hit in the *other* trie first
            fuzzy = []
            if in_en ^ in_es:
                other = es_trie if in_en else en_trie
                fuzzy = difflib.get_close_matches(tok, other.keys(),
                                                  n=1, cutoff=fuzzy_cutoff)
                if debug:
                    print(f"[FUZZY] tok={tok!r}, fuzzy_match={fuzzy}")
                if not fuzzy:
                    # truly unambiguous by lexicon
                    w["lang"] = "en" if in_en else "es"
                    w["lang_confidence"] = 1.0
                    if debug:
                        print(f"[ASSIGN] {tok!r} → {w['lang']} (lexicon)\n")
                    continue
                # else: fall through to tie-breaker & audio but keep `fuzzy[0]` in mind

            # 4.2) Zipf‐sigmoid tie‐breaker
            freq_en = zipf_frequency(tok, "en")
            freq_es = zipf_frequency(tok, "es")
            diff    = freq_en - freq_es
            p_en    = 1.0 / (1.0 + 10 ** (-diff))
            p_es    = 1.0 - p_en
            if debug:
                print(f"[ZIPF] tok={tok!r}, p_en={p_en:.2f}, p_es={p_es:.2f}")

            if p_en >= margin:
                w["lang"], w["lang_confidence"] = "en", p_en
                if debug:
                    print(f"[ASSIGN] {tok!r} → en (zipf)\n")
                continue
            elif p_es >= margin:
                w["lang"], w["lang_confidence"] = "es", p_es
                if debug:
                    print(f"[ASSIGN] {tok!r} → es (zipf)\n")
                continue

            # 4.3) Audio‐based fallback
            if debug:
                print(f"[AUDIO] fallback for {tok!r}")
            start = max(0.0, w["start"] - pad)
            end   = min(total_dur, w["end"] + pad)
            clip  = audio[int(start * sr) : int(end * sr)]
            if clip.size == 0:
                w["lang"], w["lang_confidence"] = None, 0.0
                if debug:
                    print(f"[AUDIO] {tok!r} → clip empty\n")
                continue

            # wrap into [1, T]
            wav = torch.from_numpy(clip).unsqueeze(0)

            # (a) feature extractor (STFT/FBANK…)
            fe_mod = next((m for n,m in model.mods.items() if "compute" in n), None)
            feats  = fe_mod(wav) if fe_mod else wav

            # (b) encoder/embedding
            enc_mod = next(
                (m for n,m in model.mods.items()
                    if ("embed" in n or "encoder" in n) and "compute" not in n),
                None
            )
            if enc_mod is None:
                raise RuntimeError("Can't find encoder module in model.mods")
            embeds = enc_mod(feats)

            # (c) classification head → logits
            head_mod = next(
                (m for n,m in model.mods.items() if "classif" in n or "proj" in n),
                None
            )
            if head_mod is None:
                raise RuntimeError("Can't find classification head in model.mods")
            logits = head_mod(embeds).squeeze(0)   # → [N_class]

            # (d) temperature‐scaled softmax → probabilities
            probs_arr = torch.softmax(logits / temp, dim=-1).cpu().numpy()
            # sometimes it comes as [[…]] shape
            if probs_arr.ndim == 2 and probs_arr.shape[0] == 1:
                probs_arr = probs_arr[0]

            # (e) recover label texts & codes
            le      = model.hparams.label_encoder
            ind2lab = le.ind2lab
            labels  = [ind2lab[i] for i in range(len(probs_arr))]
            codes   = [lbl.split(":", 1)[0] for lbl in labels]

            # (f) pick
            probs_map = dict(zip(codes, probs_arr))
            p_en2     = probs_map.get("en", 0.0)
            p_es2     = probs_map.get("es", 0.0)
            if p_en2 >= p_es2:
                chosen, conf = "en", p_en2
            else:
                chosen, conf = "es", p_es2

            w["lang"], w["lang_confidence"] = chosen, float(conf)
            if debug:
                print(f"[AUDIO] {tok!r} → {chosen} ({conf:.2f})\n")

    return segments
