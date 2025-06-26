#!/usr/bin/env python3
# test_short.py
"""
Runs the full pipeline on test.m4a, then generates and compiles
a LaTeX screenplay (single speaker) into lingualign/tex/.
"""

import os
import sys
import warnings
import subprocess
from pprint import pprint

import whisperx
import torch
from lingualign.lid import init_audio_lid, annotate_segments_language
from lingualign.io import load_and_normalize
from tqdm import tqdm

# ─── suppress noisy warnings ────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=r"You are using `torch.load`")
warnings.filterwarnings("ignore", message=r"No language specified.*")
warnings.filterwarnings("ignore", message=r"audio is shorter than 30s.*")

# ─── project setup ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── paths & model params ───────────────────────────────────────────────────
audio_file = os.path.join(PROJECT_ROOT, "raw_audio", "test.m4a")
batch_size = 16
device     = "cpu"
compute    = "float32"
sr         = 16000

# ─── 1) Load WhisperX ASR model ────────────────────────────────────────────
print("▶ Loading WhisperX ASR model 'medium'…")
asr_model = whisperx.load_model("medium", device=device, compute_type=compute)

# ─── 2) Transcribe ─────────────────────────────────────────────────────────
print("▶ Loading & decoding audio via io.load_and_normalize…")
audio, file_sr = load_and_normalize(audio_file, sr=sr, mono=True)
print(f"   → got {audio.shape[0]} samples @ {file_sr} Hz")
print("▶ Transcribing test.m4a…")
result = asr_model.transcribe(audio, batch_size=batch_size)
print(f"✔ Produced {len(result['segments'])} segments")

# ─── 3) Align word-level timestamps ────────────────────────────────────────
print("▶ Aligning word-level timestamps…")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    audio,
    device,
    return_char_alignments=False
)["segments"]
print(f"✔ Aligned into {len(aligned)} segments")

# ─── 4) Speaker diarization ────────────────────────────────────────────────
print("▶ Performing speaker diarization…")
diarizer = whisperx.diarize.DiarizationPipeline(device=device)
turns    = diarizer(audio)
dia      = whisperx.assign_word_speakers(turns, {"segments": aligned})
segments = dia["segments"]
for seg in segments:
    seg_spk = seg.get("speaker", "UNK")
    for w in seg["words"]:
        if "speaker" not in w or w["speaker"] is None:
            w["speaker"] = seg_spk
print("✔ Assigned speaker labels")

# ─── 5) Per-word language ID ───────────────────────────────────────────────
print("▶ Initializing SpeechBrain LID model…")
sb_lid = init_audio_lid(device=device)

print("▶ Annotating per-word language…")
annotated_segments = annotate_segments_language(
    segments,            # WhisperX+diarization segments
    audio,               # np.ndarray already loaded @ sr=16 kHz
    sr=sr,
    en_lex_path="lexicons/english.txt",
    es_lex_path="lexicons/spanish.txt",
    model=sb_lid,        # pass the SpeechBrain classifier
    margin=0.75,         # if neither text-based score ≥ 0.75, fall back to audio LID
    temp=5.0,             # temperature for the
    debug=True
)
print("✔ Completed language annotation")

# ─── 6) Print transcript ─────────────────────────────────────────────────
print("\nFinal transcript (start–end | speaker | lang | conf | word):")
for seg in annotated_segments:
    for w in seg["words"]:
        s, e = w["start"], w["end"]
        spk  = w.get("speaker", "UNK")
        lang = w.get("lang", "-")
        conf = w.get("lang_confidence", 0.0)
        print(f"{s:.2f}–{e:.2f} | {spk:>8} | {lang:<2} {conf:.2f} | {w['word']}")

# ─── 7) Generate & compile LaTeX screenplay ────────────────────────────────

def sanitize_latex(s: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = s.replace("“", "``").replace("”", "''").replace('"', "''")
    return s

tex_dir  = os.path.join(PROJECT_ROOT, "tex")
os.makedirs(tex_dir, exist_ok=True)
tex_file = os.path.join(tex_dir, "transcript_short.tex")

lines = [
    r"\documentclass{screenplay}",
    r"\title{test.m4a Transcript}",
    r"\author{Single Speaker}",
    r"\begin{document}",
    r"\maketitle",
    r"\centretitle{test.m4a}",
    ""
]

for seg in tqdm(annotated_segments, desc="Rendering LaTeX"):
    role   = "Narrator"
    tokens = []
    for w in seg["words"]:
        tok = sanitize_latex(w["word"])
        if w.get("lang") == "es":
            tok = r"\textbf{" + tok + "}"
        tokens.append(tok)
    lines += [
        r"\begin{dialogue}{" + role + r"}",
        " ".join(tokens),
        r"\end{dialogue}",
        ""
    ]

lines.append(r"\end{document}")

with open(tex_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✔ Written LaTeX to {tex_file}")

print("▶ Compiling PDF (pdflatex)…")
subprocess.run(
    ["pdflatex", "-interaction=batchmode", os.path.basename(tex_file)],
    cwd=tex_dir,
    check=True
)
subprocess.run(
    ["pdflatex", "-interaction=batchmode", os.path.basename(tex_file)],
    cwd=tex_dir,
    check=True
)
print(f"✔ PDF generated at {os.path.join(tex_dir, 'transcript_short.pdf')}")