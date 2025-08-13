# lingualign/lang_registry.py
from __future__ import annotations
from typing import Dict, Set, Iterable, Optional
import importlib

ALIASES_3_TO_1: Dict[str, str] = {
    'eng':'en','spa':'es','por':'pt','fra':'fr','fre':'fr','deu':'de','ger':'de','ita':'it','rus':'ru',
    'zho':'zh','cmn':'zh','arb':'ar','hbs':'sh','hrv':'hr','srp':'sr','bos':'bs','kat':'ka','ell':'el',
    'grc':'el','hin':'hi','ben':'bn','urd':'ur','fas':'fa','pes':'fa','tur':'tr','ukr':'uk','nld':'nl',
    'ron':'ro','rum':'ro','swe':'sv','nno':'nn','nob':'no','nor':'no','isl':'is','dan':'da','fin':'fi',
    'tha':'th','vie':'vi','jpn':'ja','kor':'ko','heb':'he','hau':'ha','amh':'am','mlg':'mg','tgl':'tl',
    'ceb':'ceb','yid':'yi','yue':'zh','wuu':'zh','pan':'pa','pus':'ps','snd':'sd','sin':'si','slk':'sk',
    'slv':'sl','hye':'hy','aze':'az','eus':'eu','glg':'gl','lav':'lv','lit':'lt','mkd':'mk','tam':'ta',
    'tel':'te','kan':'kn','mal':'ml','guj':'gu','mar':'mr','nep':'ne','lao':'lo','khm':'km','msa':'ms',
    'mya':'my','prs':'fa','ckb':'ku','kur':'ku','tuk':'tk','tjk':'tg','uzb':'uz','kaz':'kk','tat':'tt',
    'sqi':'sq','bel':'be','bul':'bg','cat':'ca','est':'et','fao':'fo','oci':'oc','gla':'gd','gle':'ga',
    'glv':'gv','ltz':'lb','lin':'ln','yor':'yo','ibo':'ig','sna':'sn','swa':'sw','som':'so','tir':'ti',
    'uzn':'uz','zhx':'zh',
}

NAME_ALIASES: Dict[str,str] = {
    'english':'en','spanish':'es','castilian':'es','portuguese':'pt','french':'fr','german':'de','italian':'it',
    'russian':'ru','mandarin':'zh','mandarin chinese':'zh','chinese':'zh','arabic':'ar','hindi':'hi','bengali':'bn',
    'urdu':'ur','persian':'fa','farsi':'fa','turkish':'tr','ukrainian':'uk','dutch':'nl','romanian':'ro','swedish':'sv',
    'norwegian':'no','norwegian nynorsk':'nn','icelandic':'is','danish':'da','finnish':'fi','thai':'th','vietnamese':'vi',
    'japanese':'ja','korean':'ko','hebrew':'he','yiddish':'yi','greek':'el','georgian':'ka','kazakh':'kk','azerbaijani':'az',
    'armenian':'hy','basque':'eu','galician':'gl','catalan':'ca','slovak':'sk','slovenian':'sl','czech':'cs','polish':'pl',
    'hungarian':'hu','croatian':'hr','serbian':'sr','bosnian':'bs','albanian':'sq','bulgarian':'bg','macedonian':'mk',
    'latvian':'lv','lithuanian':'lt','welsh':'cy','irish':'ga','scots gaelic':'gd','occitan':'oc','faroese':'fo','maltese':'mt',
    'malay':'ms','indonesian':'id','filipino':'tl','tagalog':'tl','cebuano':'ceb','javanese':'jv','lao':'lo','khmer':'km',
    'burmese':'my','punjabi':'pa','panjabi':'pa','pashto':'ps','pushto':'ps','sindhi':'sd','sinhala':'si','kannada':'kn',
    'tamil':'ta','telugu':'te','malayalam':'ml','gujarati':'gu','marathi':'mr','nepali':'ne','swahili':'sw','somali':'so',
    'yoruba':'yo','lingala':'ln','hausa':'ha','tatar':'tt','tajik':'tg','uzbek':'uz','luxembourgish':'lb','maori':'mi','breton':'br',
}

def normalize_lang(s: str) -> str:
    if not s: return s
    k = s.strip().lower().replace('_','-')
    if '-' in k and len(k.split('-',1)[0]) in (2,3):
        k = k.split('-',1)[0]
    if len(k)==3 and k in ALIASES_3_TO_1: return ALIASES_3_TO_1[k]
    if k in NAME_ALIASES: return NAME_ALIASES[k]
    if len(k)==2: return k
    return k

def _probe_whisper_langs() -> Set[str]:
    langs: Set[str] = set()
    try:
        whisper = importlib.import_module('whisper')
        if hasattr(whisper,'tokenizer') and hasattr(whisper.tokenizer,'LANGUAGES'):
            for name, code in whisper.tokenizer.LANGUAGES.items():
                langs.add(code)
        langs.add('en')
    except Exception:
        langs.update({'en','es','fr','de','it','pt','ru','zh','ar','ja','ko'})
    return langs

def _probe_align_langs() -> Set[str]:
    langs: Set[str] = set()
    try:
        wx_align = importlib.import_module('whisperx.alignment')
        for attr in dir(wx_align):
            if attr.startswith('DEFAULT_ALIGN_MODELS'):
                model_map = getattr(wx_align, attr)
                langs.update(set(model_map.keys()))
    except Exception:
        langs.update({'en','es','fr','de','it','pt','ru','zh','ar','ja'})
    return langs

def capabilities(include_runtime: bool=True, runtime_aligns: Optional[Set[str]]=None, runtime_whisper: Optional[Set[str]]=None) -> Dict[str,Set[str]]:
    if include_runtime:
        asr = runtime_whisper or _probe_whisper_langs()
        align = runtime_aligns or _probe_align_langs()
        lid = set()
    else:
        asr = {'en','es','fr','de','it','pt','ru','zh','ar','ja','ko'}
        align = {'en','es','fr','de','it','pt','ru','zh','ar','ja'}
        lid = set()
    return {'asr':asr,'align':align,'lid':lid}

def effective_language_set(scope: str, user_whitelist: Optional[Iterable[str]], caps: Dict[str,Set[str]], fallback_if_empty: Optional[Set[str]]=None) -> Set[str]:
    wl = set(normalize_lang(x) for x in user_whitelist) if user_whitelist else None
    if scope not in {'any','asr','intersection'}:
        scope = 'intersection'
    if scope == 'any':
        base = caps['lid'] if caps['lid'] else (caps['asr'] | caps['align'])
    elif scope == 'asr':
        base = caps['asr']
    else:
        base = (caps['lid'] if caps['lid'] else caps['asr']) & caps['asr'] & caps['align']
    eff = base if wl is None else (base & wl)
    if not eff:
        eff = fallback_if_empty or (wl or base or {'en'})
    return eff
