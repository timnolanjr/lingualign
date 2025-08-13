#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
try:
    from wordfreq import top_n_list
except Exception as e:
    raise SystemExit('This tool requires `wordfreq`. Install it with `pip install wordfreq`.') from e
ALIASES_3_TO_1={'eng':'en','spa':'es','por':'pt','fra':'fr','fre':'fr','deu':'de','ger':'de'}
NAME_ALIASES={'english':'en','spanish':'es','portuguese':'pt','french':'fr','german':'de','italian':'it'}
def normalize_lang(s:str)->str:
    if not s:return s
    k=s.strip().lower().replace('_','-')
    if '-' in k and len(k.split('-',1)[0]) in (2,3):k=k.split('-',1)[0]
    if len(k)==3 and k in ALIASES_3_TO_1:return ALIASES_3_TO_1[k]
    if k in NAME_ALIASES:return NAME_ALIASES[k]
    if len(k)==2:return k
    return k
def main():
    ap=argparse.ArgumentParser('Build lexicons with wordfreq')
    ap.add_argument('--lang',required=True)
    ap.add_argument('--top',type=int,default=50000)
    ap.add_argument('--min-len',type=int,default=1)
    ap.add_argument('--out',type=str,default='lexicons')
    ap.add_argument('--wordlist',type=str,default='large',choices=['small','large'])
    args=ap.parse_args()
    langs=[normalize_lang(x) for x in args.lang.split(',') if x.strip()]
    outdir=Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for lg in langs:
        words=top_n_list(lg, args.top, wordlist=args.wordlist)
        words=[w.lower().strip() for w in words if len(w)>=args.min_len and not w.isspace()]
        seen=set(); uniq=[]
        for w in words:
            if w not in seen: seen.add(w); uniq.append(w)
        outpath=outdir/(lg+'.txt')
        with outpath.open('w', encoding='utf-8') as f:
            for w in uniq: f.write(w+'\n')
        print('\u2713', lg, '->', outpath)
if __name__=='__main__':
    main()
