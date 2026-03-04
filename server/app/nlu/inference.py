"""ONNX runtime inference wrapper for JointBERT intent+slot model."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class NLUResult:
    """Result from JointBERT inference."""
    intent: str
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)


class NLUInference:
    """ONNX-based JointBERT inference for intent classification + slot extraction."""

    def __init__(self, model_dir: str, max_seq_len: int = 50):
        model_path = Path(model_dir)
        onnx_path = model_path / "model_int8.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"JointBERT ONNX model not found at {onnx_path}")

        # Load label lists
        with open(model_path / "intent_labels.json") as f:
            self._intent_labels: list[str] = json.load(f)
        with open(model_path / "slot_labels.json") as f:
            self._slot_labels: list[str] = json.load(f)

        # Load tokenizer
        self._tokenizer = BertTokenizerFast.from_pretrained(str(model_path / "tokenizer"))
        self._max_seq_len = max_seq_len

        # ONNX session setup (matches SmartTurn pattern)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        active = self._session.get_providers()
        logger.info(f"NLUInference loaded from {onnx_path} | providers={active}")

    def predict(self, text: str) -> NLUResult:
        """Run inference on a single utterance. Returns intent, confidence, and extracted slots."""
        t0 = time.perf_counter()

        # Tokenize
        encoding = self._tokenizer(
            text,
            max_length=self._max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # ONNX forward pass — build inputs based on what the model expects
        input_names = {i.name for i in self._session.get_inputs()}
        feed = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names and "token_type_ids" in encoding:
            feed["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)

        outputs = self._session.run(["intent_logits", "slot_logits"], feed)

        intent_logits = outputs[0][0]   # (num_intents,)
        slot_logits = outputs[1][0]     # (seq_len, num_slots)

        # Intent: softmax → best label + confidence
        intent_probs = _softmax(intent_logits)
        intent_id = int(np.argmax(intent_probs))
        confidence = float(intent_probs[intent_id])
        intent = self._intent_labels[intent_id]

        # Slots: argmax per token → decode BIO spans
        slot_ids = np.argmax(slot_logits, axis=-1)   # (seq_len,)
        slots = self._decode_slots(text, encoding, slot_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"NLU predict: '{text[:60]}' → {intent} ({confidence:.3f}) slots={slots} in {elapsed_ms:.1f}ms")

        return NLUResult(intent=intent, confidence=confidence, slots=slots)

    def _decode_slots(self, text: str, encoding, slot_ids: np.ndarray) -> dict[str, str]:
        """Decode BIO tag predictions back to {slot_type: "extracted text"} dict."""
        tokens = self._tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        word_ids = encoding.word_ids(batch_index=0)

        # Group slot predictions by word
        word_slot_tags: dict[int, str] = {}
        prev_word_id = None
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_word_id:
                # First subtoken of this word — use its slot prediction
                tag = self._slot_labels[slot_ids[i]] if slot_ids[i] < len(self._slot_labels) else "O"
                word_slot_tags[wid] = tag
            prev_word_id = wid

        # Split original text into words and reconstruct spans from BIO tags
        words = text.split()
        slots: dict[str, str] = {}
        current_type = None
        current_words: list[str] = []

        for wid in range(len(words)):
            tag = word_slot_tags.get(wid, "O")

            if tag.startswith("B-"):
                # Save previous span if any
                if current_type and current_words:
                    slots[current_type] = " ".join(current_words)
                current_type = tag[2:]
                current_words = [words[wid]]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_words.append(words[wid])
            else:
                # O tag or mismatched I- tag — close current span
                if current_type and current_words:
                    slots[current_type] = " ".join(current_words)
                current_type = None
                current_words = []

        # Close final span
        if current_type and current_words:
            slots[current_type] = " ".join(current_words)

        return slots


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
