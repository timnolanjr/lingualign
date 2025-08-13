
# LinguaAlign
**Multilingual, code‑switching captions — per‑word LID with WhisperX & VoxLingua107**

LinguaAlign is a production‑oriented pipeline for audio/video where speakers **switch languages mid‑sentence**. It keeps the dialogue **as spoken** instead of forcing everything into one language or silently rewriting via translation.

- **ASR**: Whisper/WhisperX with word‑level timestamps
- **LID**: per‑word language tags via a 3‑stage cascade (**Lexicon → Zipf (wordfreq) → VoxLingua107**)
- **Any‑N languages**: `--languages en,es,ar,...` (not hard‑coded pairs)
- **Exports**: TXT, Markdown, SRT, WebVTT (via SRT), LaTeX (PDF optional)

> **Vision:** Multilingual closed captioning barely exists as a service today. LinguaAlign aims to power a **free web app** so creators and language instructors can publish **faithful, code‑switching captions** — improving accessibility, pedagogy, and authenticity.

---

## Table of contents
1. [Motivation](#motivation)
2. [Who is this for?](#who-is-this-for)
3. [Key features](#key-features)
4. [Language coverage](#language-coverage)
5. [Install](#install)
6. [Quick start](#quick-start)
7. [Generate lexicons](#generate-lexicons)
8. [CLI usage](#cli-usage)
9. [Exports](#exports)
10. [How it works](#how-it-works)
11. [Web app vision](#web-app-vision)
12. [Design principles](#design-principles)
13. [Performance tips](#performance-tips)
14. [Troubleshooting](#troubleshooting)
15. [Project layout](#project-layout)
16. [FAQ](#faq)
17. [License](#license)

---

## Motivation

### Speech isn’t monolingual
In classrooms, livestreams, podcasts, and community meetings, people **code‑switch** naturally: *“Ok, empiezo con el setup de Docker y luego vemos el **datasets** folder.”* One utterance may contain **two or more languages**. That’s not an edge case — that’s normal communication.

### Single‑language funnels distort reality
Most captioning pipelines (and many “auto” modes) **funnel output to one language**. Even when detectors sense multiple languages, results are stored or edited **per segment/file**, not per word. Consequences:
- **Mistranscriptions** when a foreign‑language word is forced into the dominant language’s spelling.
- **Lost nuance** for idioms, proper names, code terms, and quotations.
- **Poor pedagogy** for language learning: students can’t see where the switch happened.
- **Compliance issues** in legal/medical/archival contexts where the transcript must reflect **what was actually said**.

**LinguaAlign preserves the dialogue as it happens** — tagging each **word** with a language and aligning it honestly.

---

## Who is this for?
- **Creators & streamers** who mix languages and want accessible, accurate captions.
- **Language instructors** who need **code‑switch‑aware** materials for teaching and assessment.
- **Archivists & journalists** requiring faithful transcripts (no auto‑translation drift).
- **Researchers** studying multilingual discourse and code‑switching.

---

## Key features
- **Any‑N languages**: pass a whitelist; only the relevant lexicons are loaded.
- **Per‑language decoding**: try forced decodes for top‑K candidates; select by ASR logprob for stability.
- **3‑stage Language ID**: lexicon membership → Zipf heuristic → VoxLingua107 audio LID fallback.
- **Exports**: TXT / MD / SRT / LaTeX with per‑language highlighting. More output formats available upon request.

---

## Language coverage
- **ASR (Whisper multilingual)**: English + **96 non‑English** languages (97 total; matches your installed Whisper mapping).
- **LID (VoxLingua107)**: **107 languages** via `speechbrain/lang-id-voxlingua107-ecapa`.
- **Alignment (WhisperX)**: subset with align models; we detect support at runtime and fall back gracefully.

**Recommendation:** `--lang-scope intersection` to restrict predictions to languages supported by **all** active stages.

---

## Install
Requires Python 3.9+ and FFmpeg on PATH.

```bash
pip install whisperx speechbrain soundfile wordfreq marisa-trie
# (Optional) Install PyTorch with CUDA/ROCm for GPU acceleration.
```

---

## Quick start
1) **Build lexicons** for your languages:
```bash
python tools/build_lexicon.py --lang en,es,ar --top 60000 --out lexicons
```
Or generate for **all Whisper multilingual languages**:
```bash
python tools/build_whisper_lexicons_all.py --top 60000 --out lexicons --include-english
# If needed for Javanese: --alias jw:jv
```

2) **Run the pipeline**:
```bash
python -m lingualign.pipeline path/to/audio.mp3   -o results   --languages en,es,ar   --lang-scope intersection   --asr-topk 2   --highlight-langs ar   --formats plain,md,srt,tex
```

Outputs are placed in `results/`.

---

## Generate lexicons
- **Specific languages:** `tools/build_lexicon.py` (accepts names, 2‑letter, and many 3‑letter codes).  
- **All Whisper languages:** `tools/build_whisper_lexicons_all.py` (pulls the mapping from your installed Whisper/faster‑whisper).

Normalization makes `eng/english/EN‑US` → `en`, etc. Runtime loading only pulls the files needed for your whitelist.

---

## CLI usage
```bash
python -m lingualign.pipeline AUDIO   [-o OUTDIR] [--formats plain,md,srt,tex]   [--sr 16000] [--lexicon-dir lexicons]   [--languages en,es,ar] [--lang-scope any|asr|intersection]   [--asr-language en] [--asr-topk 2]   [--device-asr cpu|cuda:0] [--device-align cpu|cuda:0]   [--batch-size 16] [--highlight-langs ar] [--debug]
```

- `--languages`: comma‑separated whitelist (any N). If omitted, we infer from runtime caps.
- `--lang-scope`: `intersection` (recommended), `asr`, or `any` (union of detected sets).
- `--asr-language`: force a single language (fastest when known).
- `--asr-topk`: number of forced‑language trials to score and select.
- `--highlight-langs`: emphasize those languages in exports.
- `--lexicon-dir`: directory containing `<code>.txt` lexicons.

---

## Exports
- **TXT**: 1 line per segment. Highlighted languages → `***word***`.
- **Markdown**: bullet per segment. Same highlighting.
- **SRT**: standard captions; highlighted tokens italic+bold via HTML tags.
- **LaTeX**: printable transcript; highlighted words → `\textbf{\textit{...}}`. Optional PDF if `pdflatex` is installed.

> Tip: SRT is widely supported; convert to **WebVTT** for the web if needed (`ffmpeg` or `srt2vtt`).

---

## How it works
### 1) Candidate languages
You supply `--languages`, or we derive plausible sets from runtime capabilities (Whisper, WhisperX aligners, VoxLingua107). `--lang-scope intersection` recommends languages supported by **all** active stages.

### 2) ASR with per‑language trials
We can **force** Whisper to decode with `language=ℓ` for each candidate (up to `--asr-topk`), score by average token logprob, and pick the winner. This reduces drift in mixed‑language content.

### 3) Per‑word LID cascade
1. **Lexicon** membership (exact/fuzzy) for decisive cases.  
2. **Zipf** tie‑break using `wordfreq.zipf_frequency` over the candidate set.  
3. **Audio LID** (optional) with VoxLingua107 to resolve hard cases.

### 4) Alignment & exports
If a WhisperX aligner exists for the segment’s language, we use it to refine word timings; otherwise we keep Whisper timestamps and **label the absence** of an align model. Exports then render with per‑language emphasis.

---

## Web app vision
We plan a **free web app** on top of LinguaAlign to make code‑switching captions accessible to everyone — especially **creators** and **language instructors**.

**MVP goals**
- Upload audio/video, pick target languages (or detect), run pipeline.
- Interactive **caption editor** with per‑word **language colors** and timeline scrub.
- Export **SRT/WebVTT/ASS/JSON/LaTeX**; preserve language tags; toggle highlights.
- “Teacher mode”: quickly isolate words in L2, auto‑generate drills from highlighted ranges.

**Suggested architecture (open to contributions)**
- **Backend**: FastAPI (Python) + worker queue (RQ/Celery) calling LinguaAlign. Persistent storage to S3‑compatible blob + Postgres.
- **Frontend**: React + a waveform/timeline component; color‑coding by language; shortcut‑driven edits.
- **Auth & quotas**: email/GitHub login; fair‑use rate limits to keep it **free**.
- **Privacy**: short‑lived object storage; user‑scoped encryption; easy **Delete** per job.

If you’d like to help build the hosted app, open an issue with your interest/skills.

---

## Design principles
- **Fidelity over fluency**: never rewrite content into a single language.
- **Evidence‑driven**: prefer strong lexicon hits; be transparent when falling back.
- **Minimal coupling**: keep components composable; optional dependencies stay optional.
- **Honest gaps**: if a model is missing (e.g., no aligner), say so and keep timestamps.

---

## Performance tips
- Prefer GPU (`--device-asr cuda:0`, `--device-align cuda:0`) for long recordings.
- Keep `--languages` tight; fewer candidates → faster forced decodes.
- Build lexicons with `--top 60k` (or higher for rich morphology) and `--min-len 2`.
- Chunk very long recordings upstream to keep memory stable.

---

## Troubleshooting
- **wordfreq unsupported**: use `--alias src:dst` (e.g., `jw:jv`) or reduce `--top`.
- **No align model**: output still works with Whisper timestamps; try a different align build or `--lang-scope asr`.
- **Unexpected language tags**: verify lexicons exist for all whitelist items; check `--lang-scope` and runtime caps.
- **FFmpeg missing**: install FFmpeg and ensure it’s on PATH.

---

## Project layout
```
lingualign/
  __init__.py
  lang_registry.py      # normalization + runtime capability probing
  lexicon.py            # loads only the needed <lang>.txt (en, eng, english, ...)
  lid.py                # 3-stage per-word LID cascade
  exporters.py          # txt, md, srt, tex (optional PDF)
  pipeline.py           # CLI wrapper & I/O

tools/
  build_lexicon.py                  # build lexicons for specific languages
  build_whisper_lexicons_all.py     # build for all Whisper multilingual languages
```

---

## FAQ
**Why not rely on automatic language detection alone?**  
Auto detection typically picks **one language per file/segment**. We need **per‑word** tags and must preserve multiple languages **within** a sentence — without forcing spelling/translation.

**Is translation supported?**  
No. LinguaAlign is about **preserving what was said**. You can translate downstream.

**How are `en` vs `eng` handled?**  
We normalize many code/name variants (639‑3 and common names) to consistent 2‑letter codes wherever possible.

---

## License
MIT for this codebase. Whisper/WhisperX/SpeechBrain and any downloaded models are under their respective licenses.

