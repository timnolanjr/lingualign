# lingualign/vosk_lid.py
"""
Dual-pass Vosk-based bilingual transcription for Lingualign:
- Transcribes entire audio with English and Spanish models
- Merges results by timestamp and confidence
- Groups adjacent tokens into language segments
"""
import json
import numpy as np
from typing import List, Dict
from vosk import Model, KaldiRecognizer
from lingualign.io import load_and_normalize

# Cache Vosk models in memory
default_models: Dict[str, Model] = {}

def load_vosk_model(model_path: str) -> Model:
    """
    Load or return a cached Vosk Model.
    """
    if model_path not in default_models:
        default_models[model_path] = Model(model_path)
    return default_models[model_path]


def transcribe_vosk(
    path: str,
    model_path: str,
    frames_per_chunk: int = 4000
) -> List[Dict]:
    """
    Transcribe any file via io.load_and_normalize + Vosk model.
    Returns a list of {word, start, end, conf} dicts.
    """
    # 1) Load audio
    audio, sr = load_and_normalize(path, sr=16000, mono=True)
    # 2) Convert to 16-bit PCM bytes
    pcm = (audio * 32767).astype(np.int16).tobytes()
    # 3) Initialize recognizer
    model = load_vosk_model(model_path)
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(True)

    # 4) Feed chunks
    results: List[Dict] = []
    bytes_per_chunk = frames_per_chunk * 2
    for i in range(0, len(pcm), bytes_per_chunk):
        chunk = pcm[i : i + bytes_per_chunk]
        if rec.AcceptWaveform(chunk):
            res = json.loads(rec.Result())
            results.extend(res.get("result", []))
    # 5) Final partials
    final = json.loads(rec.FinalResult())
    results.extend(final.get("result", []))
    return results


def merge_vosk_results(
    en_res: List[Dict],
    es_res: List[Dict]
) -> List[Dict]:
    """
    Merge two time-ordered Vosk results (en and es) into one list,
    resolving overlaps by confidence.
    Each dict gains a 'lang' key.
    """
    merged: List[Dict] = []
    i, j = 0, 0
    while i < len(en_res) and j < len(es_res):
        e = en_res[i]
        s = es_res[j]
        if e['end'] <= s['start']:
            e['lang'] = 'en'
            merged.append(e)
            i += 1
        elif s['end'] <= e['start']:
            s['lang'] = 'es'
            merged.append(s)
            j += 1
        else:
            # overlap
            if e.get('conf', 0) >= s.get('conf', 0):
                e['lang'] = 'en'
                merged.append(e)
                i += 1
            else:
                s['lang'] = 'es'
                merged.append(s)
                j += 1
    # append remainders
    for k in range(i, len(en_res)):
        en_res[k]['lang'] = 'en'
        merged.append(en_res[k])
    for k in range(j, len(es_res)):
        es_res[k]['lang'] = 'es'
        merged.append(es_res[k])
    # sort by start
    merged.sort(key=lambda w: w['start'])
    return merged


def group_lang_segments(tokens: List[Dict]) -> List[Dict]:
    """
    Combine adjacent tokens with identical 'lang' into segments.
    Returns list of {start, end, lang, text}.
    """
    if not tokens:
        return []
    segments: List[Dict] = []
    cur = {
        'start': tokens[0]['start'],
        'end':   tokens[0]['end'],
        'lang':  tokens[0]['lang'],
        'text':  tokens[0]['word']
    }
    for tok in tokens[1:]:
        if tok['lang'] == cur['lang']:
            cur['end'] = tok['end']
            cur['text'] += ' ' + tok['word']
        else:
            segments.append(cur)
            cur = {
                'start': tok['start'],
                'end':   tok['end'],
                'lang':  tok['lang'],
                'text':  tok['word']
            }
    segments.append(cur)
    return segments


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python -m lingualign.vosk_lid <audio_path> <en_model> <es_model>')
        sys.exit(1)
    audio_path = sys.argv[1]
    en_model   = sys.argv[2]
    es_model   = sys.argv[3]

    # 1) transcribe both
    en_res = transcribe_vosk(audio_path, en_model)
    es_res = transcribe_vosk(audio_path, es_model)
    # 2) merge tokens
    tokens = merge_vosk_results(en_res, es_res)
    # 3) group into segments
    segments = group_lang_segments(tokens)

    # print segments
    for seg in segments:
        print(f"{seg['start']:.2f}-{seg['end']:.2f} [{seg['lang']}] {seg['text']}")
