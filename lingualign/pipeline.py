# lingualign/pipeline.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .exporters import to_plain, to_markdown, to_srt, to_tex

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import whisperx
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

try:
    from speechbrain.inference.classifiers import EncoderClassifier
    HAS_SPEECHBRAIN = True
except Exception:
    HAS_SPEECHBRAIN = False

from .lang_registry import normalize_lang, capabilities, effective_language_set
from .lid import annotate_segments_language, LIDConfig
from .exporters import to_plain, to_markdown, to_srt

def _load_audio(path: str, sr: int) -> np.ndarray:
    if sf is None:
        raise RuntimeError('soundfile not installed. `pip install soundfile`')
    x, file_sr = sf.read(path, dtype='float32', always_2d=False)
    if file_sr != sr:
        ratio = sr / float(file_sr)
        new_len = int(len(x) * ratio)
        idx = np.linspace(0, len(x) - 1, new_len).astype(np.int64)
        x = x[idx]
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return x

def _transcribe_whisperx(audio_path: str, device_asr: str, device_align: str, batch_size: int, forced_language: Optional[str]) -> Dict:
    if not HAS_WHISPERX:
        raise RuntimeError('whisperx not installed. `pip install whisperx`')
    model = whisperx.load_model('medium', device_asr, compute_type='float16' if 'cuda' in device_asr else 'float32')
    audio = whisperx.load_audio(audio_path)
    tx_kwargs = {'batch_size': batch_size}
    if forced_language:
        tx_kwargs['language'] = forced_language
    result = model.transcribe(audio, **tx_kwargs)
    lang = result.get('language')
    try:
        align_model, metadata = whisperx.load_align_model(language=lang, device=device_align)
        result['segments'] = whisperx.align(result['segments'], align_model, metadata, audio, device_align)['segments']
    except Exception:
        pass
    return result

def decode_with_language_candidates(audio_path: str, candidates: List[str], device_asr: str, device_align: str, batch_size: int, asr_topk: int) -> Dict:
    if not HAS_WHISPERX or not candidates:
        return _transcribe_whisperx(audio_path, device_asr, device_align, batch_size, None)
    scored = []
    for lg in candidates[:max(1, asr_topk)]:
        try:
            out = _transcribe_whisperx(audio_path, device_asr, device_align, batch_size, lg)
            logs = [s.get('avg_logprob', -5.0) for s in out.get('segments', [])]
            score = float(np.mean(logs)) if logs else -5.0
            scored.append((score, out))
        except Exception:
            continue
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    return _transcribe_whisperx(audio_path, device_asr, device_align, batch_size, None)

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser('lingualign: N-language ASR+LID pipeline')
    ap.add_argument('audio', help='Path to audio file')
    ap.add_argument('-o','--outdir', default='results', help='Output directory')
    ap.add_argument('-f','--formats', default='plain,md,srt', help='Comma-separated: plain,md,srt')
    ap.add_argument('--sr', type=int, default=16000, help='LID audio sample rate')
    ap.add_argument('--lexicon-dir', type=str, default='lexicons', help='Directory of <lang>.txt files')
    ap.add_argument('--languages', type=str, default=None, help='Comma-separated whitelist, e.g. en,es,ar')
    ap.add_argument('--lang-scope', type=str, default='intersection', choices=['any','asr','intersection'], help='Candidate set scope')
    ap.add_argument('--highlight-langs', type=str, default=None, help='Comma-separated langs to emphasize in outputs')
    ap.add_argument('--asr-language', type=str, default=None, help='Force Whisper language (skip candidate search)')
    ap.add_argument('--lid-topk', type=int, default=2, help='(reserved for future segment-level redecoding)')
    ap.add_argument('--asr-topk', type=int, default=2, help='Try top-K candidate languages with forced decoding')
    ap.add_argument('--device-asr', type=str, default='cpu')
    ap.add_argument('--device-align', type=str, default='cpu')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args(argv)

    audio_path = args.audio
    outdir = Path(args.outdir)

    caps = capabilities(include_runtime=True)
    user_list = [normalize_lang(s) for s in args.languages.split(',')] if args.languages else None

    sb_model = None
    if HAS_SPEECHBRAIN:
        try:
            sb_model = EncoderClassifier.from_hparams(
                source='speechbrain/lang-id-voxlingua107-ecapa',
                savedir='pretrained_models/lang-id-voxlingua107-ecapa',
                run_opts={'device': args.device_asr},
            )
            try:
                import numpy as _np
                labels = sb_model.hparams.label_encoder.decode_ndim(_np.arange(sb_model.hparams.label_encoder.num_labels))
                vox_codes = set(l.split(':',1)[0].strip().lower() for l in labels)
                caps['lid'] = {normalize_lang(c) for c in vox_codes}
            except Exception:
                pass
        except Exception:
            sb_model = None

    candidates = effective_language_set(scope=args.lang_scope, user_whitelist=user_list, caps=caps, fallback_if_empty={'en'})
    candidates = sorted(candidates)

    if args.asr_language:
        forced = normalize_lang(args.asr_language)
        result = decode_with_language_candidates(audio_path, [forced], args.device_asr, args.device_align, args.batch_size, asr_topk=1)
        chosen_langs = [forced]
    else:
        result = decode_with_language_candidates(audio_path, candidates, args.device_asr, args.device_align, args.batch_size, asr_topk=args.asr_topk)
        chosen_langs = candidates

    segments = result.get('segments', [])
    audio_arr = _load_audio(audio_path, args.sr)

    lid_cfg = LIDConfig(languages=chosen_langs, lexicon_dir=args.lexicon_dir, margin=0.75, fuzzy_cutoff=0.85, pad=0.1, sr=args.sr, debug=args.debug)
    segments = annotate_segments_language(segments, audio_arr, sb_model, lid_cfg)

    fmts = [x.strip().lower() for x in args.formats.split(',') if x.strip()]
    hl = [normalize_lang(x) for x in args.highlight_langs.split(',')] if args.highlight_langs else None

    if "plain" in fmts:
        to_plain(segments, audio_path, outdir, highlight_langs=hl)
    if "md" in fmts or "markdown" in fmts:
        to_markdown(segments, audio_path, outdir, highlight_langs=hl)
    if "srt" in fmts:
        to_srt(segments, audio_path, outdir, highlight_langs=hl)
    if "tex" in fmts:
        to_tex(segments, audio_path, outdir, compile_pdf=False, title=Path(audio_path).stem, highlight_langs=hl)

    print(f"✓ Wrote outputs to {outdir.resolve()}")

if __name__ == '__main__':
    main()
