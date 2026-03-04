"""JointBERT model — dual-head intent classification + BIO slot tagging."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class JointBERT(nn.Module):
    """BERT/DistilBERT backbone with intent classification head ([CLS]) and slot tagging head (all tokens)."""

    def __init__(
        self,
        backbone: str,
        num_intents: int,
        num_slots: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self._has_pooler = hasattr(self.bert, "pooler") and self.bert.pooler is not None
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_intents),
        )
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_slots),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use pooler_output if available (BERT), else [CLS] token (DistilBERT)
        if self._has_pooler:
            cls_output = outputs.pooler_output
        else:
            cls_output = outputs.last_hidden_state[:, 0, :]
        intent_logits = self.intent_head(cls_output)                  # (B, num_intents)
        slot_logits = self.slot_head(outputs.last_hidden_state)       # (B, seq_len, num_slots)
        return intent_logits, slot_logits
