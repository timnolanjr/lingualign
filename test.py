#!/usr/bin/env python3
# test.py

import os
import sys
import warnings
from pprint import pprint

import whisperx
from lingualign.lid import annotate_segments_language
from tqdm import tqdm

# ─── Suppress non‐critical warnings ─────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", message=r"No language specified.*")
warnings.filterwarnings("ignore", message=r"audio is shorter than 30s.*")

# ─── Ensure project root on PYTHONPATH ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── Configuration ─────────────────────────────────────────────────────────
audio_file = os.path.join(
    PROJECT_ROOT,
    "raw_audio",
    "soundcloudaud.com_Complete Spanish, Track 2 - Language Transfer, The Thinking Method.mp3"
)
batch_size = 16

# ─── 1) Load WhisperX ASR model “medium” with CUDA/FP16 → CPU/FP32 fallback ─
device, compute = "cuda", "float16"
try:
    print(f"▶ Loading WhisperX ASR model 'medium' on {device}/{compute}…")
    asr_model = whisperx.load_model(
        "medium", device=device, compute_type=compute
    )
except ValueError as e:
    msg = str(e).lower()
    if "float16" in msg or "cuda support" in msg:
        print(f"⚠️  {e} — falling back to CPU/FP32")
        device, compute = "cpu", "float32"
        asr_model = whisperx.load_model(
            "medium", device=device, compute_type=compute
        )
    else:
        raise

# ─── 2) Transcribe ─────────────────────────────────────────────────────────
print("▶ Transcribing…")
audio = whisperx.load_audio(audio_file)
result = asr_model.transcribe(audio, batch_size=batch_size)
print(f"✔ Produced {len(result['segments'])} segments")

# ─── 3) Align (skip file‐level LID) ────────────────────────────────────────
print("▶ Aligning word‐level timestamps…")
asr_lang = result["language"]
align_model, metadata = whisperx.load_align_model(
    language_code=asr_lang, device=device
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

# ─── 4) Diarize via WhisperX’s in‐repo checkpoint ─────────────────────────
print("▶ Performing speaker diarization…")
diarizer = whisperx.diarize.DiarizationPipeline(device=device)
turns = diarizer(audio)
dia = whisperx.assign_word_speakers(turns, {"segments": aligned})
annotated_segments = dia["segments"]
print("✔ Assigned speaker labels")

# ─── 5) Load WhisperX LID model (tiny) & annotate ─────────────────────────
print("▶ Loading WhisperX LID model 'tiny'…")
lid_model = whisperx.load_model(
    "tiny",
    device=device,
    compute_type=compute,
    language=asr_lang
)
print("▶ Annotating per-word language…")
annotated_segments = annotate_segments_language(
    annotated_segments,
    audio_file,
    sr=16000,
    model=lid_model
)
print("✔ Completed language annotation")

# ─── 6) Print final transcript to console ─────────────────────────────────
print("\nFinal transcript (start–end | speaker | lang | confidence | word):")
for seg in annotated_segments:
    for w in seg["words"]:
        s, e = w["start"], w["end"]
        spk  = w.get("speaker", "UNK")
        lang = w.get("lang", "-")
        conf = w.get("lang_confidence", 0.0)
        print(f"{s:.2f}–{e:.2f} | {spk:>8} | {lang:<2} {conf:.2f} | {w['word']}")

# ─── 7) Generate LaTeX screenplay with bolded Spanish words ────────────────
print("\n▶ Generating LaTeX screenplay…")

# 7a) helper to escape LaTeX special chars and smart quotes
def sanitize_latex(s: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # convert curly quotes and straight " to proper LaTeX quotes
    s = s.replace("“", "``").replace("”", "''")
    s = s.replace('"', "''")
    return s

# 7b) map speaker IDs to roles
role_map = {
    "SPEAKER_00": "Student",
    "SPEAKER_01": "Teacher"
}

# 7c) prepare output directory
tex_dir = os.path.join(PROJECT_ROOT, "tex")
os.makedirs(tex_dir, exist_ok=True)
tex_path = os.path.join(tex_dir, "transcript.tex")

# 7d) build the LaTeX lines
tex_lines = [
    r"\documentclass{screenplay}",
    r"\title{Language Transfer, The Thinking Method}",
    r"\author{Teacher \& Student}",
    r"\begin{document}",
    r"\maketitle",
    r"\centretitle{Complete Spanish, Track 2 -- Language Transfer}",
    ""
]

for seg in tqdm(annotated_segments, desc="Rendering LaTeX"):
    spk = seg.get("speaker", "UNK")
    role = role_map.get(spk, spk)
    # gather tokens, bold Spanish
    tokens = []
    for w in seg["words"]:
        tok = sanitize_latex(w["word"])
        if w.get("lang") == "es":
            tokens.append(r"\textbf{" + tok + "}")
        else:
            tokens.append(tok)
    line = " ".join(tokens)
    tex_lines.append(r"\begin{dialogue}{" + role + r"}")
    tex_lines.append(line)
    tex_lines.append(r"\end{dialogue}")
    tex_lines.append("")

tex_lines.append(r"\end{document}")

# 7e) write out
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

print(f"✔ Written LaTeX screenplay to {tex_path}")
print(f"  You can compile it via:")
print(f"    cd {tex_dir} && pdflatex transcript.tex")
