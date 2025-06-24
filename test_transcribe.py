# test_transcribe.py

import sys
# Strip any extra CLI args so unittest won’t choke
sys.argv = sys.argv[:1]

import unittest
import subprocess
import re
from pathlib import Path

class TestTranscribeCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent
        cls.audio = cls.project_root / "raw_audio" / "test.m4a"
        assert cls.audio.exists(), f"Test file not found: {cls.audio}"
        cls.python = sys.executable
        cls.module = "lingualign.transcribe"
        # regex for per‐segment lines
        cls.line_re = re.compile(r'^\d+\.\d{2}s–\d+\.\d{2}s: (en|es) ".*"$')

    def test_cli_runs_and_formats(self):
        cmd = [
            self.python,
            "-m", self.module,
            str(self.audio),
            "--model_size", "tiny",
            "--frame_ms", "20",
            "--padding_s", "0.1",
        ]
        proc = subprocess.run(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Must exit cleanly
        self.assertEqual(proc.returncode, 0, f"Non-zero exit: {proc.stderr}")

        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        # Find only the per-segment lines (those matching our regex)
        segment_lines = [l for l in lines if self.line_re.match(l)]
        self.assertTrue(segment_lines, "No per-segment output lines")

        # Validate each segment line
        for ln in segment_lines:
            self.assertRegex(ln, self.line_re, f"Bad format: {ln}")

        # Optionally, also check that we saw the full-transcript header
        self.assertIn(
            "Full transcript:", lines,
            "Missing ‘Full transcript:’ section"
        )

if __name__ == "__main__":
    unittest.main()
