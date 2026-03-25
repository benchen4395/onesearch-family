import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Optional, Union
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class JointSDFTTrainer(CustomSeq2SeqTrainer):
    """
    Joint Self-Distillation Fine-Tuning Trainer.

    joint mode: Single model, two forward passes (teacher sees full input, student sees reduced input).
                Gradients from all losses are accumulated on the same set of parameters.
    ema mode:   Teacher is an independent copy (no gradients), updated via EMA from the student.
    """

    def __init__(
        self,
        sdft_mode: str = "joint",
        kl_weight: float = 0.1,
        distill_temperature: float = 1.0,
        teacher_ce_weight: float = 1.0,
        ema_decay: float = 0.999,
        teacher_model: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sdft_mode = sdft_mode
        self.kl_weight = kl_weight
        self.distill_temperature = distill_temperature
        self.teacher_ce_weight = teacher_ce_weight
        self.ema_decay = ema_decay
        self.teacher_model = teacher_model

        if self.sdft_mode == "ema" and self.teacher_model is None:
            raise ValueError("EMA mode requires a teacher_model. Ensure it is created in the workflow.")
        if self.sdft_mode == "joint" and self.teacher_model is not None:
            logger.warning_rank0("Joint mode does not use teacher_model; the provided one will be ignored.")
            self.teacher_model = None

        self._sdft_metrics = {
            "student_ce": 0.0,
            "teacher_ce": 0.0,
            "kl_loss": 0.0,
            "step_count": 0,
        }

        logger.info_rank0(
            f"[Joint SDFT] mode='{self.sdft_mode}' | kl_weight={self.kl_weight} | "
            f"temperature={self.distill_temperature} | teacher_ce_weight={self.teacher_ce_weight}"
            + (f" | ema_decay={self.ema_decay}" if self.sdft_mode == "ema" else "")
        )

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        If inputs contain teacher fields -> SDFT logic.
        Otherwise -> fall back to standard SFT (used during evaluation/predict).
        """
        teacher_input_ids = inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = inputs.pop("teacher_attention_mask", None)
        teacher_labels = inputs.pop("teacher_labels", None)

        if teacher_input_ids is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        teacher_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "labels": teacher_labels,
        }

        if self.sdft_mode == "joint":
            loss = self._compute_loss_joint(model, inputs, teacher_inputs)
        elif self.sdft_mode == "ema":
            loss = self._compute_loss_ema(model, inputs, teacher_inputs)
        else:
            raise ValueError(f"Unknown sdft_mode: {self.sdft_mode}")

        if return_outputs:
            with torch.no_grad():
                student_outputs = model(**inputs)
            return (loss, student_outputs)
        return loss

    def _compute_loss_joint(self, model, student_inputs, teacher_inputs):
        """Joint mode: teacher forward with no_grad to avoid DDP + gradient_checkpointing conflicts."""
        with torch.no_grad():
            teacher_outputs = model(**teacher_inputs)
            teacher_ce_loss = teacher_outputs.loss
            teacher_logits = teacher_outputs.logits

        student_outputs = model(**student_inputs)
        student_ce_loss = student_outputs.loss
        student_logits = student_outputs.logits

        kl_loss = self._compute_kl_loss_batched(
            student_logits, teacher_logits,
            student_inputs.get("labels"), teacher_inputs.get("labels"),
        )

        total_loss = student_ce_loss + self.kl_weight * kl_loss
        self._log_metrics(student_ce_loss, teacher_ce_loss, kl_loss)
        return total_loss

    def _compute_loss_ema(self, model, student_inputs, teacher_inputs):
        """EMA mode: teacher is an independent copy, inference only (no gradients)."""
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_inputs["input_ids"],
                attention_mask=teacher_inputs["attention_mask"],
                labels=teacher_inputs["labels"],
            )
            teacher_ce_loss = teacher_outputs.loss
            teacher_logits = teacher_outputs.logits

        student_outputs = model(**student_inputs)
        student_ce_loss = student_outputs.loss
        student_logits = student_outputs.logits

        kl_loss = self._compute_kl_loss_batched(
            student_logits, teacher_logits,
            student_inputs.get("labels"), teacher_inputs.get("labels"),
        )

        total_loss = student_ce_loss + self.kl_weight * kl_loss
        self._log_metrics(student_ce_loss, teacher_ce_loss, kl_loss)
        return total_loss

    @torch.no_grad()
    def _ema_update_teacher(self):
        """teacher = ema_decay * teacher + (1 - ema_decay) * student"""
        student_params = dict(self.model.named_parameters())
        for name, teacher_param in self.teacher_model.named_parameters():
            if name in student_params:
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_params[name].data, alpha=1.0 - self.ema_decay
                )

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.sdft_mode == "ema" and self.teacher_model is not None:
            self._ema_update_teacher()
        return loss

    def _compute_kl_loss_batched(self, student_logits, teacher_logits, student_labels, teacher_labels):
        """
        Batched Forward KL: KL(P_teacher || P_student).

        Teacher and student sequences differ in length (instruction portion varies),
        but the output tokens (labels != -100) are identical. We tail-align valid
        output logits per sample, pad to the max aligned length in the batch, then
        compute softmax + kl_div in a single batched call.
        """
        batch_size = student_logits.size(0)
        device = student_logits.device
        temperature = self.distill_temperature

        s_valid_mask = (student_labels != IGNORE_INDEX)
        t_valid_mask = (teacher_labels != IGNORE_INDEX)

        s_valid_counts = s_valid_mask.sum(dim=1)
        t_valid_counts = t_valid_mask.sum(dim=1)
        aligned_counts = torch.min(s_valid_counts, t_valid_counts)
        max_aligned_len = aligned_counts.max().item()

        if max_aligned_len == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        vocab_size = student_logits.size(-1)
        s_aligned = torch.zeros(batch_size, max_aligned_len, vocab_size, device=device)
        t_aligned = torch.zeros(batch_size, max_aligned_len, vocab_size, device=device)
        valid_mask = torch.zeros(batch_size, max_aligned_len, device=device)

        for b in range(batch_size):
            n = aligned_counts[b].item()
            if n == 0:
                continue

            s_logits_valid = student_logits[b][s_valid_mask[b]]
            t_logits_valid = teacher_logits[b][t_valid_mask[b]]

            s_aligned[b, :n] = s_logits_valid[-n:]
            t_aligned[b, :n] = t_logits_valid[-n:]
            valid_mask[b, :n] = 1.0

        t_probs = F.softmax(t_aligned / temperature, dim=-1)
        s_log_probs = F.log_softmax(s_aligned / temperature, dim=-1)

        kl_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
        kl_per_token = kl_per_token * valid_mask

        total_valid = valid_mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        kl_loss = kl_per_token.sum() / total_valid
        kl_loss = kl_loss * (temperature ** 2)
        return kl_loss

    def _compute_kl_loss(self, student_logits, teacher_logits, student_labels, teacher_labels):
        """Per-sample KL divergence (legacy fallback)."""
        batch_size = student_logits.size(0)
        device = student_logits.device
        temperature = self.distill_temperature
        kl_losses = []

        for b in range(batch_size):
            s_valid = student_labels[b] != IGNORE_INDEX
            t_valid = teacher_labels[b] != IGNORE_INDEX

            s_logits = student_logits[b][s_valid]
            t_logits = teacher_logits[b][t_valid]

            n = min(s_logits.size(0), t_logits.size(0))
            if n == 0:
                continue

            s_logits = s_logits[-n:]
            t_logits = t_logits[-n:]

            t_probs = F.softmax(t_logits / temperature, dim=-1)
            s_log_probs = F.log_softmax(s_logits / temperature, dim=-1)
            kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
            kl = kl * (temperature ** 2)
            kl_losses.append(kl)

        if not kl_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(kl_losses).mean()

    def _log_metrics(self, student_ce, teacher_ce, kl_loss):
        self._sdft_metrics["student_ce"] += student_ce.item()
        self._sdft_metrics["teacher_ce"] += teacher_ce.item()
        self._sdft_metrics["kl_loss"] += kl_loss.item()
        self._sdft_metrics["step_count"] += 1

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._sdft_metrics["step_count"] > 0:
            n = self._sdft_metrics["step_count"]
            logs["sdft/student_ce"] = round(self._sdft_metrics["student_ce"] / n, 6)
            logs["sdft/teacher_ce"] = round(self._sdft_metrics["teacher_ce"] / n, 6)
            logs["sdft/kl_loss"] = round(self._sdft_metrics["kl_loss"] / n, 6)
            logs["sdft/mode"] = 0.0 if self.sdft_mode == "joint" else 1.0
            self._sdft_metrics = {
                "student_ce": 0.0,
                "teacher_ce": 0.0,
                "kl_loss": 0.0,
                "step_count": 0,
            }
        super().log(logs, *args, **kwargs)