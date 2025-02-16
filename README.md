# MultilingualTranscriptMaker

When presented with a multilingual audio file, most transcript applications default to a single language and automatically translate non-target-language segments. **MultilingualTranscriptMaker** is a tool that produces a **multilingual output** that retains the original language diversity. This is especially useful for creating language learning materials.

**MultilingualTranscriptMaker** was built with the following goals in mind:
- **Preserve Language Diversity:** Keep all the different languages present in an audio file.
- **Flexible Output Options:** Enable future support for both multilingual transcripts and single-language outputs that highlight off-language segments (see [Feature Roadmap](#feature-roadmap)).

## Current Features

- **Bilingual English \& Spanish Transcription:** As a first pass, proof-of-concept, I've started with transcription of audio files that contain English and Spanish (my particular use case).

## Feature Roadmap

- **Multilingual Transcription:** Accurately transcribe audio files containing multiple languages without default auto-translation.
- **Flexible Output Formats:** Maintain the original language context in the transcript.

## Requirements

- Python 3.8+
- [OpenAI Whisper](https://github.com/openai/whisper) for audio transcription

## Installation

**Clone the repository:**

   ```bash
   git clone https://github.com/timnolanjr/MultilingualTranscriptMaker.git
   cd MultilingualTranscriptMaker 
```

## Shoutouts

A special thanks to OpenAI for *actually* being true to their name for once with Whisper, and [cainky/SoundcloudDownloader](https://github.com/cainky/SoundcloudDownloader) for a helpful tool!


