#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
try:
    from wordfreq import top_n_list
except Exception as e:
    raise SystemExit('This tool requires `wordfreq`. Install it with `pip install wordfreq`.') from e
def get_whisper_lang_map():
    langs=None
    try:
        import whisper
        if hasattr(whisper,'tokenizer') and hasattr(whisper.tokenizer,'LANGUAGES'):
            langs=whisper.tokenizer.LANGUAGES
    except Exception:
        pass
    if langs is None:
        try:
            from faster_whisper import tokenizer as fw_tok
            langs=getattr(fw_tok,'LANGUAGES',None)
        except Exception:
            pass
    if langs is None:
        raise SystemExit('Install `openai-whisper` or `faster-whisper`.')
    return langs
DEFAULT_CODE_ALIASES={'jw':'jv'}
def normalize_code(code:str)->str:
    code=code.strip().lower()
    if '-' in code: code=code.split('-',1)[0]
    return code
def main():
    ap=argparse.ArgumentParser('Build lexicons for Whisper multilingual languages')
    ap.add_argument('--top',type=int,default=60000)
    ap.add_argument('--min-len',type=int,default=1)
    ap.add_argument('--out',type=str,default='lexicons')
    ap.add_argument('--wordlist',type=str,default='large',choices=['small','large'])
    ap.add_argument('--include-english',dest='include_english',action='store_true')
    ap.add_argument('--no-english',dest='include_english',action='store_false')
    ap.set_defaults(include_english=False)
    ap.add_argument('--alias',action='append',default=[])
    args=ap.parse_args()
    code_aliases=dict(DEFAULT_CODE_ALIASES)
    for a in args.alias:
        if ':' in a:
            src,dst=a.split(':',1); code_aliases[src.strip().lower()]=dst.strip().lower()
    langs_map=get_whisper_lang_map()
    codes=set(langs_map.values())
    if args.include_english: codes.add('en')
    outdir=Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    built=[]; skipped=[]
    for code in sorted(codes):
        wf_code=code_aliases.get(code, code)
        wf_code=normalize_code(wf_code)
        try:
            words=top_n_list(wf_code, args.top, wordlist=args.wordlist)
        except Exception as e:
            skipped.append((code, str(e))); continue
        words=[w.lower().strip() for w in words if len(w)>=args.min_len and not w.isspace()]
        seen=set(); uniq=[]
        for w in words:
            if w not in seen: seen.add(w); uniq.append(w)
        outpath=outdir/(normalize_code(code)+'.txt')
        with outpath.open('w', encoding='utf-8') as f:
            for w in uniq: f.write(w+'\n')
        built.append((code, len(uniq)))
    print('Built', len(built), 'lexicons into', outdir.resolve())
    if skipped:
        print('Skipped:')
        for code, reason in skipped:
            print(' ', code, '-', reason)
if __name__=='__main__':
    main()
