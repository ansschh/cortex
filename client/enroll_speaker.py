"""Speaker enrollment script — record your voice to create a reference embedding."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from client.audio.mic import MicStream
from client.audio.speaker_verify import SpeakerVerifier


def main():
    print("=" * 50)
    print("  NOVA — Speaker Enrollment")
    print("=" * 50)
    print()

    threshold = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.65"))
    verifier = SpeakerVerifier(threshold=threshold)

    print("[1/3] Loading speaker verification model...")
    verifier.initialize()

    if not verifier.is_available:
        print("ERROR: Speaker verification model could not be loaded.")
        print("Make sure speechbrain and torch are installed:")
        print("  pip install speechbrain torchaudio torch")
        sys.exit(1)

    print("[2/3] Model loaded. Preparing microphone...")
    mic = MicStream()
    mic.start()

    print()
    print("Ready to record. You will record 3 samples of ~5 seconds each.")
    print("Speak naturally — as if giving a command to the assistant.")
    print()

    all_segments = []

    for i in range(3):
        input(f"Press ENTER to start recording sample {i + 1}/3...")
        print(f"  Recording sample {i + 1}... speak now!")

        chunks = []
        start = time.time()
        duration = 5.0

        while time.time() - start < duration:
            chunk = mic.get_chunk(timeout=0.05)
            if chunk is not None:
                chunks.append(chunk)
            time.sleep(0.01)

        if not chunks:
            print("  WARNING: No audio captured. Check your microphone.")
            continue

        audio = np.concatenate(chunks)
        energy = np.abs(audio).mean()
        print(f"  Captured {len(audio) / 16000:.1f}s of audio (energy={energy:.0f})")

        if energy < 100:
            print("  WARNING: Very low audio energy. Speak louder or check mic.")

        # Split into 2-second segments
        seg_len = 16000 * 2
        for j in range(0, len(audio) - seg_len, seg_len):
            all_segments.append(audio[j:j + seg_len])

        print(f"  Sample {i + 1} captured.")
        print()

    mic.stop()

    if len(all_segments) < 3:
        print("ERROR: Not enough audio segments for enrollment.")
        print("Please try again and speak more clearly.")
        sys.exit(1)

    print(f"[3/3] Computing speaker embedding from {len(all_segments)} segments...")
    success = verifier.enroll(all_segments)

    if success:
        print()
        print("=" * 50)
        print("  Enrollment COMPLETE!")
        print(f"  Saved to: {os.path.abspath('data/speaker_enrollment/reference_embedding.npy')}")
        print(f"  Verification threshold: {threshold}")
        print("=" * 50)
        print()
        print("Test it by running the client and speaking a command.")
    else:
        print("ERROR: Enrollment failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
