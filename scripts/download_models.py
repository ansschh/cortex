#!/usr/bin/env python3
"""Download all ML models required by NOVA.

Usage:
    python scripts/download_models.py              # Download all models
    python scripts/download_models.py --server-only # Server models only
    python scripts/download_models.py --client-only # Client models only
    python scripts/download_models.py --cpu-only    # Skip GPU-specific models
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_whisper(model_size: str = "large-v3"):
    """Download faster-whisper model for local STT."""
    print(f"\n[1/6] Downloading faster-whisper {model_size}...")
    try:
        from faster_whisper import WhisperModel
        # This triggers the download to the default cache
        print(f"  Loading {model_size} (this downloads ~3GB on first run)...")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        del _model
        print(f"  ✓ faster-whisper {model_size} ready")
    except ImportError:
        print("  ✗ faster-whisper not installed. Run: pip install faster-whisper>=1.0.0")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def download_smolvlm():
    """Download SmolVLM-500M for local vision."""
    print("\n[2/6] Downloading SmolVLM-500M-Instruct...")
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"
        print(f"  Downloading processor...")
        AutoProcessor.from_pretrained(model_name)
        print(f"  Downloading model (~1GB)...")
        AutoModelForImageTextToText.from_pretrained(model_name)
        print(f"  ✓ SmolVLM-500M ready")
    except ImportError:
        print("  ✗ transformers not installed. Run: pip install transformers")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def download_speechbrain():
    """Download SpeechBrain speaker verification model."""
    print("\n[3/6] Downloading SpeechBrain speaker verification model...")
    model_dir = ensure_dir(DATA / "speechbrain_model")
    try:
        from speechbrain.pretrained import SpeakerRecognition
        print("  Downloading ECAPA-TDNN model...")
        SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
        )
        print(f"  ✓ SpeechBrain model saved to {model_dir}")
    except ImportError:
        print("  ✗ speechbrain not installed. Run: pip install speechbrain")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def download_sentence_transformers():
    """Download sentence-transformers model for memory embeddings."""
    print("\n[4/6] Downloading sentence-transformers (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        del model
        print("  ✓ Sentence transformers model ready")
    except ImportError:
        print("  ✗ sentence-transformers not installed. Run: pip install sentence-transformers")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def download_silero_vad():
    """Download Silero VAD model."""
    print("\n[5/6] Downloading Silero VAD...")
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        del model, utils
        print("  ✓ Silero VAD ready")
    except ImportError:
        print("  ✗ torch not installed. Run: pip install torch")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def download_smart_turn():
    """Check/download SmartTurn ONNX model."""
    print("\n[6/6] Checking SmartTurn v3 model...")
    model_dir = DATA / "models" / "smart_turn"
    model_path = model_dir / "smart_turn_v3.2.onnx"
    if model_path.exists():
        print(f"  ✓ SmartTurn model found at {model_path}")
    else:
        ensure_dir(model_dir)
        print(f"  ✗ SmartTurn model not found at {model_path}")
        print("    This model is custom-trained. Contact the project maintainer for the weights.")


def main():
    parser = argparse.ArgumentParser(description="Download NOVA ML models")
    parser.add_argument("--server-only", action="store_true", help="Only download server-side models")
    parser.add_argument("--client-only", action="store_true", help="Only download client-side models")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU-specific configurations")
    args = parser.parse_args()

    print("=" * 60)
    print("NOVA — Model Downloader")
    print("=" * 60)

    # Ensure data directories exist
    ensure_dir(DATA / "models" / "jointbert")
    ensure_dir(DATA / "models" / "piper")
    ensure_dir(DATA / "models" / "smart_turn")
    ensure_dir(DATA / "speechbrain_model")
    ensure_dir(DATA / "speaker_enrollment")

    if not args.server_only:
        # Client models
        download_whisper()
        download_speechbrain()
        download_silero_vad()
        download_smart_turn()

    if not args.client_only:
        # Server models
        download_smolvlm()
        download_sentence_transformers()

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNotes:")
    print("  - JointBERT NLU model: train with `python scripts/train_jointbert.py`")
    print("  - Piper TTS model: place .onnx + .json in data/models/piper/")
    print("  - SmartTurn model: contact maintainer for custom weights")


if __name__ == "__main__":
    main()
