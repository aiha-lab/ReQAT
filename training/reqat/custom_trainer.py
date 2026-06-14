import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Union

from modelopt.torch.quantization.plugins import QATSFTTrainer


class SEMTrainer(QATSFTTrainer):
    """QAT trainer with Selective Entropy Minimization (SEM).

    Applies entropy minimization loss weighted by lambda_sem on the
    lowest-entropy tokens (bottom entropy_threshold fraction), encouraging
    the model to be confident on tokens it is already confident about.
    """

    def __init__(self, *args, entropy_threshold=0.25, lambda_sem=0.01, sem_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_sem = lambda_sem
        self.entropy_threshold = entropy_threshold
        self.sem_only = sem_only

    def masked_ce(self, outputs, labels, num_items_in_batch=None):
        logits = outputs.logits  # [B, T, V]
        vocab_size = logits.size(-1)
        logits = logits.float().view(-1, vocab_size)  # [B*T, V]

        # Pre-shift labels
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous().view(-1)  # [B*T]
        shift_labels = shift_labels.to(logits.device)

        probs = F.softmax(logits, dim=-1)  # [B*T, V]
        entropy = -(probs * (probs + 1e-12).log()).sum(-1)  # [B*T]
        valid_mask = shift_labels != -100

        # Cross-entropy loss
        if self.sem_only:
            ce_loss = 0.0
        else:
            ce_loss = F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(ce_loss.device)
            ce_loss = ce_loss / num_items_in_batch

        # Selective entropy minimization on low-entropy tokens
        if valid_mask.any():
            valid_entropy = entropy[valid_mask]

            with torch.no_grad():
                k = max(int(valid_entropy.numel() * self.entropy_threshold), 1)
                threshold_entropy = -torch.topk(-valid_entropy, k).values.min()
                min_entropy = valid_entropy.min()
                sem_weights = torch.clamp(
                    1.0 - (valid_entropy - min_entropy) / (threshold_entropy - min_entropy + 1e-12),
                    min=0.0,
                )

            sem_loss = (valid_entropy * sem_weights).sum() * self.lambda_sem / num_items_in_batch
        else:
            sem_loss = 0.0

        return ce_loss + sem_loss

    def compute_loss(self, model, inputs, *args, **kwargs):
        if not model.training:
            _compute_loss_func = self.compute_loss_func
            self.compute_loss_func = None

        loss = super().compute_loss(model, inputs, *args, **kwargs)

        if not model.training:
            self.compute_loss_func = _compute_loss_func

        return loss

    def train(self, *args, **kwargs):
        self.compute_loss_func = self.masked_ce
        return super().train(*args, **kwargs)
