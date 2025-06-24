# test_vad.py

"""
End-to-end smoke test for our VAD module.
Loads an audio file, runs get_speech_turns, and prints each detected segment.
"""

import sys
from lingualign.vad import get_speech_turns

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_vad.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Running VAD on: {audio_path}\n")

    turns = get_speech_turns(audio_path, frame_ms=30.0, padding_s=0.1)
    if not turns:
        print("No speech detected.")
        return

    print("Detected speech turns:")
    for i, (start, end) in enumerate(turns, 1):
        dur = end - start
        print(f"{i:2d}. {start:.2f}s → {end:.2f}s  (duration {dur:.2f}s)")

if __name__ == "__main__":
    main()
