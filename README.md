# MultilingualTranscriptMaker

**MultilingualTranscriptMaker** is a tool designed to create transcripts from audio files while preserving multiple languages. Unlike other transcript applications that default to a single language (and automatically translate off-language segments), this tool produces a **multilingual output** that retains the original language diversity. This is especially useful for creating language learning materials.

## Motivation

Online transcript applications typically prompt the user to select a single language. If an audio file contains multiple languages, these tools often auto-translate the off-language audio into the target language, which loses valuable context and nuances.

**MultilingualTranscriptMaker** was built with these goals in mind:
- **Preserve Language Diversity:** Keep all the different languages present in an audio file.
- **Improve Language Learning:** Generate transcripts that allow learners to identify and compare different language segments.
- **Flexible Output Options:** Enable future support for both multilingual transcripts and single-language outputs that highlight off-language segments.

## Features

- **Multilingual Transcription:** Accurately transcribe audio files containing multiple languages without default auto-translation.
- **Flexible Output Formats:** Maintain the original language context in the transcript.
- **Easy Integration:** Designed to integrate with tools like [OpenAI Whisper](https://github.com/openai/whisper) for transcription.

## Requirements

- Python 3.8+
- [OpenAI Whisper](https://github.com/openai/whisper) for audio transcription

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/timnolanjr/MultilingualTranscriptMaker.git
   cd MultilingualTranscriptMaker

