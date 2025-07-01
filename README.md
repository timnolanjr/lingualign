# Lingualign

**Lingualign** is an end-to-end transcription toolkit designed to preserve the integrity of multilingual audio streams. While standard transcription tools default to one language, Lingualign was built to preserve the integrity of mixed-language audio content like bilingual podcasts, interviews, and songs. It was created to supplement comprehensible-input language lessons, giving learners clear, context-preserving transcripts of authentic materials. It is a tool multilingual educators and creators can use to automatically caption multilingual content.
---

## Pipeline Overview

Lingualign processes audio in four stages:

1. **Transcribe & Align**  
   Uses [WhisperX](https://github.com/m-bain/whisperX/tree/main) to generate a transcript with precise word-level timecodes.

2. **Diarize**  
   WhisperX applies PyAnnote under the hood to label “who spoke when,” so you can distinguish speakers.

3. **Language ID**  
   Runs [SpeechBrain](https://github.com/speechbrain/speechbrain) to tag each word by language (e.g., English, Spanish).

4. **Export**  
   Produces multiple formats for your needs:  
   - **Plain text** (`.txt`) for quick reading  
   - **Markdown** (`.md`) for easy annotation  
   - **SRT** (`.srt`) for subtitles  
   - **Screenplay-style PDF** (`.tex` + `.pdf`) for polished handouts  

---

## Current Features

- Hard-coded to work for English and Spanish audio only for now...
- No GPU access currently, so I'd love to hear about its performance!

---

## Roadmap

1. **Broader language support**  
   - Integrate additional lexicons (French, German, Mandarin, Arabic, etc.)  
   - Automatic model selection based on file metadata or user flag

3. **Enhanced exports**  
   - Full transcript translated entirely into 2+ languages
   - Interactive HTML transcripts with clickable speaker/language filters  
   - Subtitle packages for YouTube, TikTok, and broadcast standards (host this as a web service?)


## Thanks

This project stands on the shoulders of giants. Big thanks to:
- OpenAI for Whisper (and for actually being open...)
- m-bain for WhisperX, which makes this project possible

---

## License

MIT License — see [LICENSE](LICENSE) for details.  

