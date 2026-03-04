#!/usr/bin/env python3
"""Train JointBERT (intent + slot) on Amazon MASSIVE dataset, export to ONNX + INT8.

Usage:
    python scripts/train_jointbert.py

Outputs to data/models/jointbert/:
    model_int8.onnx     — INT8 quantized ONNX model
    intent_labels.json  — list of 60 intent names
    slot_labels.json    — list of BIO slot labels
    tokenizer/          — saved BertTokenizerFast
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Add project root to path so we can import the model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.app.nlu.model import JointBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKBONE = "distilbert-base-uncased"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "jointbert"
MAX_SEQ_LEN = 50
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MASSIVEJointDataset(Dataset):
    """Wraps a MASSIVE split for joint intent+slot training."""

    def __init__(self, hf_split, tokenizer, intent2id, slot2id, max_len=MAX_SEQ_LEN):
        self.examples = hf_split
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.slot2id = slot2id
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["utt"].split()                    # pre-tokenized words
        ner_tags = ex["annot_utt"]                    # annotated utterance string

        # Parse BIO tags from annotated utterance
        slot_labels = self._parse_bio_tags(ner_tags, words)

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align BIO tags to subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_slots = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_slots.append(-100)  # special tokens
            elif wid != prev_word_id:
                # First subtoken of a word: use the word's slot label
                if wid < len(slot_labels):
                    aligned_slots.append(self.slot2id.get(slot_labels[wid], self.slot2id.get("O", 0)))
                else:
                    aligned_slots.append(self.slot2id.get("O", 0))
            else:
                aligned_slots.append(-100)  # continuation subtokens
            prev_word_id = wid

        intent_id = self.intent2id[ex["intent"]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "intent_label": torch.tensor(intent_id, dtype=torch.long),
            "slot_labels": torch.tensor(aligned_slots, dtype=torch.long),
        }

    @staticmethod
    def _parse_bio_tags(annot_utt: str, words: list[str]) -> list[str]:
        """Parse MASSIVE annotated utterance into BIO tag list.

        MASSIVE format: "wake me up at [time : seven am] please"
        → words: ["wake", "me", "up", "at", "seven", "am", "please"]
        → tags:  ["O",    "O",  "O",  "O",  "B-time", "I-time", "O"]
        """
        tags = []
        i = 0
        in_slot = False
        slot_type = None
        annot = annot_utt

        # Walk through annotated utterance character by character
        pos = 0
        word_idx = 0
        while pos < len(annot) and word_idx < len(words):
            # Skip whitespace
            while pos < len(annot) and annot[pos] == ' ':
                pos += 1
            if pos >= len(annot):
                break

            if annot[pos] == '[':
                # Start of slot annotation: [slot_type : word1 word2 ...]
                close = annot.index(']', pos)
                inner = annot[pos + 1:close].strip()
                # Split on " : " to get slot_type and value
                parts = inner.split(' : ', 1)
                if len(parts) == 2:
                    slot_type = parts[0].strip()
                    slot_words = parts[1].strip().split()
                else:
                    # Fallback: just words, no type
                    slot_type = "UNK"
                    slot_words = inner.strip().split()

                for j, sw in enumerate(slot_words):
                    if word_idx < len(words):
                        tag = f"B-{slot_type}" if j == 0 else f"I-{slot_type}"
                        tags.append(tag)
                        word_idx += 1
                pos = close + 1
            else:
                # Regular word (outside slot annotation)
                # Find end of word
                end = pos
                while end < len(annot) and annot[end] != ' ' and annot[end] != '[':
                    end += 1
                if word_idx < len(words):
                    tags.append("O")
                    word_idx += 1
                pos = end

        # Fill remaining words with O
        while len(tags) < len(words):
            tags.append("O")

        return tags


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    logger.info("Loading MASSIVE dataset (English)...")
    jsonl_path = PROJECT_ROOT / "data" / "models" / "jointbert" / "1.0" / "data" / "en-US.jsonl"
    if not jsonl_path.exists():
        # Download if not present
        import urllib.request, tarfile
        tar_path = OUTPUT_DIR / "massive.tar.gz"
        url = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
        logger.info(f"Downloading MASSIVE 1.0 from {url}...")
        urllib.request.urlretrieve(url, str(tar_path))
        with tarfile.open(str(tar_path), "r:gz") as tar:
            en_files = [m for m in tar.getnames() if "en-US" in m]
            for m in en_files:
                tar.extract(m, str(OUTPUT_DIR))
        tar_path.unlink()
        logger.info("MASSIVE dataset downloaded and extracted.")

    # Load JSONL — single file with partition field for train/dev/test
    all_examples: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            all_examples.append(json.loads(line))

    train_split = [ex for ex in all_examples if ex["partition"] == "train"]
    val_split = [ex for ex in all_examples if ex["partition"] == "dev"]
    test_split = [ex for ex in all_examples if ex["partition"] == "test"]
    logger.info(f"Loaded {len(train_split)} train, {len(val_split)} val, {len(test_split)} test examples")

    # ---- Build label sets ----
    intent_labels = sorted(set(ex["intent"] for ex in train_split))
    intent2id = {label: i for i, label in enumerate(intent_labels)}
    logger.info(f"Intent labels: {len(intent_labels)}")

    # Collect all BIO slot labels from annotated utterances
    all_slot_labels = {"O"}
    for split in [train_split, val_split, test_split]:
        for ex in split:
            words = ex["utt"].split()
            tags = MASSIVEJointDataset._parse_bio_tags(ex["annot_utt"], words)
            all_slot_labels.update(tags)
    slot_labels = sorted(all_slot_labels)
    slot2id = {label: i for i, label in enumerate(slot_labels)}
    logger.info(f"Slot labels: {len(slot_labels)}")

    # Save labels
    with open(OUTPUT_DIR / "intent_labels.json", "w") as f:
        json.dump(intent_labels, f, indent=2)
    with open(OUTPUT_DIR / "slot_labels.json", "w") as f:
        json.dump(slot_labels, f, indent=2)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")

    # ---- Datasets & DataLoaders ----
    train_ds = MASSIVEJointDataset(train_split, tokenizer, intent2id, slot2id)
    val_ds = MASSIVEJointDataset(val_split, tokenizer, intent2id, slot2id)
    test_ds = MASSIVEJointDataset(test_split, tokenizer, intent2id, slot2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Model ----
    model = JointBERT(
        backbone=BACKBONE,
        num_intents=len(intent_labels),
        num_slots=len(slot_labels),
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Optimizer & Scheduler ----
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ---- Training ----
    best_val_intent_acc = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        intent_correct = 0
        intent_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            intent_labels_batch = batch["intent_label"].to(DEVICE)
            slot_labels_batch = batch["slot_labels"].to(DEVICE)

            intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids)

            i_loss = intent_criterion(intent_logits, intent_labels_batch)
            s_loss = slot_criterion(
                slot_logits.view(-1, slot_logits.size(-1)),
                slot_labels_batch.view(-1),
            )
            loss = i_loss + s_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = intent_logits.argmax(dim=-1)
            intent_correct += (preds == intent_labels_batch).sum().item()
            intent_total += intent_labels_batch.size(0)

        train_acc = intent_correct / intent_total
        avg_loss = total_loss / len(train_loader)

        # ---- Validation ----
        val_intent_acc, val_slot_f1 = evaluate(model, val_loader, slot_labels, slot2id)
        logger.info(
            f"Epoch {epoch}/{EPOCHS} — loss={avg_loss:.4f} train_acc={train_acc:.4f} "
            f"val_intent_acc={val_intent_acc:.4f} val_slot_f1={val_slot_f1:.4f}"
        )

        if val_intent_acc > best_val_intent_acc:
            best_val_intent_acc = val_intent_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  → New best model (intent_acc={val_intent_acc:.4f})")

    # ---- Restore best model & test ----
    model.load_state_dict(best_model_state)
    model.to(DEVICE)
    test_intent_acc, test_slot_f1 = evaluate(model, test_loader, slot_labels, slot2id)
    logger.info(f"Test — intent_acc={test_intent_acc:.4f} slot_f1={test_slot_f1:.4f}")

    # ---- Export to ONNX ----
    export_onnx(model, tokenizer, OUTPUT_DIR)

    # ---- INT8 Quantization ----
    quantize(OUTPUT_DIR)

    logger.info("Done! Model saved to " + str(OUTPUT_DIR))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, slot_labels_list, slot2id):
    """Evaluate intent accuracy and slot F1 on a data loader."""
    from seqeval.metrics import f1_score as seq_f1_score

    model.eval()
    intent_correct = 0
    intent_total = 0
    all_true_slots = []
    all_pred_slots = []

    id2slot = {v: k for k, v in slot2id.items()}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            intent_labels_batch = batch["intent_label"].to(DEVICE)
            slot_labels_batch = batch["slot_labels"]

            intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids)

            # Intent accuracy
            preds = intent_logits.argmax(dim=-1)
            intent_correct += (preds == intent_labels_batch).sum().item()
            intent_total += intent_labels_batch.size(0)

            # Slot F1 (seqeval expects list of tag sequences)
            slot_preds = slot_logits.argmax(dim=-1).cpu().numpy()
            slot_true = slot_labels_batch.numpy()

            for i in range(slot_true.shape[0]):
                true_seq = []
                pred_seq = []
                for j in range(slot_true.shape[1]):
                    if slot_true[i][j] == -100:
                        continue
                    true_seq.append(id2slot.get(slot_true[i][j], "O"))
                    pred_seq.append(id2slot.get(slot_preds[i][j], "O"))
                if true_seq:
                    all_true_slots.append(true_seq)
                    all_pred_slots.append(pred_seq)

    intent_acc = intent_correct / max(intent_total, 1)
    slot_f1 = seq_f1_score(all_true_slots, all_pred_slots) if all_true_slots else 0.0

    return intent_acc, slot_f1


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------
def export_onnx(model, tokenizer, output_dir: Path):
    """Export JointBERT to ONNX with dual outputs."""
    logger.info("Exporting to ONNX...")
    model.eval()
    model.cpu()

    dummy = tokenizer("set a timer for five minutes", return_tensors="pt",
                       max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)

    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info(f"ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")


def quantize(output_dir: Path):
    """Apply INT8 dynamic quantization to the ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    logger.info("Quantizing to INT8...")
    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"INT8 model saved: {int8_path} ({int8_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Remove unquantized model to save space
    onnx_path.unlink()
    logger.info(f"Removed unquantized model: {onnx_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
