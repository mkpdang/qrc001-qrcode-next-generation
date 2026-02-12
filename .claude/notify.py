#!/usr/bin/env python3
"""Play notification sounds for Claude Code hooks via WSLg PulseAudio."""

import math
import struct
import subprocess
import sys
import tempfile
import wave

SOUNDS = {
    "notify": {"freq": 880, "duration": 0.15, "volume": 0.4, "repeat": 2, "gap": 0.08},
    "done": {"freq": 660, "duration": 0.25, "volume": 0.5, "repeat": 1, "gap": 0},
}

SAMPLE_RATE = 44100


def generate_wav(freq: float, duration: float, volume: float, repeat: int = 1, gap: float = 0.08) -> str:
    """Generate a WAV file with a tone (optionally repeated with gaps)."""
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    w = wave.open(f.name, "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)

    for r in range(repeat):
        samples = int(SAMPLE_RATE * duration)
        for i in range(samples):
            # Apply fade in/out (5ms) to avoid clicks
            fade = min(i / (SAMPLE_RATE * 0.005), 1.0, (samples - i) / (SAMPLE_RATE * 0.005))
            val = int(32767 * volume * fade * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            w.writeframes(struct.pack("<h", val))
        # Add gap silence between repeats
        if r < repeat - 1 and gap > 0:
            gap_samples = int(SAMPLE_RATE * gap)
            w.writeframes(b"\x00\x00" * gap_samples)

    w.close()
    return f.name


def play(wav_path: str) -> None:
    """Play a WAV file using ffplay (available in WSL2 via WSLg)."""
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
        timeout=5,
        check=False,
    )


if __name__ == "__main__":
    sound = sys.argv[1] if len(sys.argv) > 1 else "notify"
    params = SOUNDS.get(sound, SOUNDS["notify"])
    wav = generate_wav(**params)
    play(wav)
