#!/usr/bin/env python3
import os

# Dynamically build the absolute path using the current working directory.
base_dir = os.getcwd()
audio_file_path = os.path.join(base_dir, "raw_audio", "test180sec.mp3")

# --- Safe Globals Setup (must be at the very top before torch-related imports) ---
from omegaconf import ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from typing import Any
from collections import defaultdict
from torch.torch_version import TorchVersion
from pyannote.audio.core.model import Introspection
from pyannote.audio.core.task import Specifications, Problem, Resolution
import torch.serialization

torch.serialization.add_safe_globals([
    ListConfig,
    ContainerMetadata,
    Metadata,
    AnyNode,
    Any,
    list,
    defaultdict,
    dict,
    int,
    TorchVersion,
    Introspection,
    Specifications,
    Problem,
    Resolution,
    # Add any additional globals as needed.
])

# --- Helper Functions ---

def transcribe_and_align(audio, forced_language, device="cpu", batch_size=4, compute_type="float32"):
    """
    Runs transcription and alignment for a forced language using large-v3.
    """
    import whisperx

    # Load the Whisper model (using large-v3).
    model = whisperx.load_model(
        "large-v3", device=device, compute_type=compute_type, language=forced_language
    )
    # Transcribe using the forced language.
    result = model.transcribe(audio, batch_size=batch_size, language=forced_language)

    # Load alignment model and align the segments.
    model_a, metadata = whisperx.load_align_model(language_code=forced_language, device=device)
    result_aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
        print_progress=True,
    )
    return result_aligned

def extract_audio_segment(audio, start, end, sr=16000):
    """
    Extracts a segment from the full audio array using start and end times (assumes sr Hz sampling rate).
    """
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    return audio[start_idx:end_idx]

def audio_language_classifier(audio_segment):
    """
    Placeholder function for audio-based language identification.
    Replace this with a call to an actual audio language identification model.
    """
    # Dummy implementation: always returns English with confidence 0.5.
    return "en", 0.5

def flatten_transcript(result):
    """
    Flattens the transcript result (segmented) into a list of word entries.
    Each entry is a dict with keys: 'word', 'start', and 'end'.
    """
    words = []
    for seg in result.get("segments", []):
        word_list = seg.get("words", [])
        if not isinstance(word_list, list):
            continue
        for word_info in word_list:
            if not isinstance(word_info, dict):
                continue
            if "word" in word_info and "start" in word_info and "end" in word_info:
                words.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                })
    return words

def merge_flat_transcripts(flat_en, flat_es, tolerance=0.25):
    """
    Merge two word lists (English and Spanish) based on overlapping start times.
    Returns a list of dicts with keys: 'start','end','english','spanish'.
    """
    merged = []
    i, j = 0, 0
    while i < len(flat_en) and j < len(flat_es):
        w_en = flat_en[i]
        w_es = flat_es[j]
        if abs(w_en["start"] - w_es["start"]) <= tolerance:
            merged.append({
                "start": min(w_en["start"], w_es["start"]),
                "end": max(w_en["end"], w_es["end"]),
                "english": w_en["word"],
                "spanish": w_es["word"],
            })
            i += 1
            j += 1
        elif w_en["start"] < w_es["start"]:
            merged.append({
                "start": w_en["start"],
                "end": w_en["end"],
                "english": w_en["word"],
                "spanish": "",
            })
            i += 1
        else:
            merged.append({
                "start": w_es["start"],
                "end": w_es["end"],
                "english": "",
                "spanish": w_es["word"],
            })
            j += 1

    # Append any leftovers
    while i < len(flat_en):
        w_en = flat_en[i]
        merged.append({
            "start": w_en["start"],
            "end": w_en["end"],
            "english": w_en["word"],
            "spanish": "",
        })
        i += 1
    while j < len(flat_es):
        w_es = flat_es[j]
        merged.append({
            "start": w_es["start"],
            "end": w_es["end"],
            "english": "",
            "spanish": w_es["word"],
        })
        j += 1

    return merged

def compare_flat_transcripts(merged_words, audio, sr=16000):
    """
    For each merged word-level entry, run language identification on the text
    from the English and Spanish fields and on the corresponding audio segment.
    """
    import langid

    for entry in merged_words:
        start, end = entry["start"], entry["end"]
        eng, spa = entry["english"], entry["spanish"]
        lang_en, conf_en = langid.classify(eng) if eng else ("", 0.0)
        lang_es, conf_es = langid.classify(spa) if spa else ("", 0.0)
        audio_seg = extract_audio_segment(audio, start, end, sr=sr)
        audio_lang, audio_conf = audio_language_classifier(audio_seg)

        print(f"[{start:.2f}-{end:.2f}]: English='{eng}' ({lang_en},{conf_en}) | "
              f"Spanish='{spa}' ({lang_es},{conf_es}) | Audio=({audio_lang},{audio_conf})")

# --- Main Execution ---
if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it in your shell.")

    import whisperx

    audio = whisperx.load_audio(audio_file_path)

    print("Transcribing/aligning English...")
    res_en = transcribe_and_align(audio, forced_language="en")

    print("Transcribing/aligning Spanish...")
    res_es = transcribe_and_align(audio, forced_language="es")

    # Flatten and merge word lists
    flat_en = flatten_transcript(res_en)
    flat_es = flatten_transcript(res_es)
    merged = merge_flat_transcripts(flat_en, flat_es)

    # Compare at word level
    print("\nMerged word-level comparison:")
    compare_flat_transcripts(merged, audio)

