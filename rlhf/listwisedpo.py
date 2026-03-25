"""
Listwise DPO trainer functions for GPT-2.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Union, Tuple, Literal


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)
    return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute per-sequence log probabilities for a decoder-only model.

    For causal LM, logits at position t predict token at position t+1,
    so we shift: logits[:, :-1, :] aligned with labels[:, 1:].
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch, seq_len) and labels (batch, seq_len) must match.")

    # Causal shift: logits[t] predicts labels[t+1]
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = labels != -100
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return (per_token_logps * loss_mask).sum(-1)


# Concatenation: merge prompt + chosen/rejected into unified sequences
def concatenated_inputs_sdpo(
    batch: Dict[str, Union[List, torch.LongTensor]],
    num_neg: int,
    device: torch.device,
) -> Dict[str, torch.LongTensor]:
    """Build concatenated batch for listwise DPO with a decoder-only model.

    Input batch keys (per sample):
      - prompt_input_ids, prompt_attention_mask          (prompt tokens)
      - chosen_input_ids, chosen_attention_mask           (prompt+response, full sequence)
      - chosen_labels                                     (prompt masked with -100, response tokens)
      - rejected_input_ids_i, rejected_attention_mask_i   (prompt+response for each negative)
      - rejected_labels_i

    Output: all (1 + num_neg) sequences stacked along dim=0, padded to max length.
    """
    bsz = batch["chosen_input_ids"].shape[0]

    # Collect all sequence lengths to find max
    all_seqs = [batch["chosen_input_ids"]]
    for i in range(num_neg):
        all_seqs.append(batch[f"rejected_input_ids_{i}"])
    max_length = max(s.shape[1] for s in all_seqs)

    def _pad(t, pad_val):
        return pad_to_length(t, max_length, pad_value=pad_val)

    # Chosen
    cat_input_ids = _pad(batch["chosen_input_ids"], 0)
    cat_attention_mask = _pad(batch["chosen_attention_mask"], 0)
    cat_labels = _pad(batch["chosen_labels"], -100)

    # Rejected (append sequentially)
    for i in range(num_neg):
        cat_input_ids = torch.cat([cat_input_ids, _pad(batch[f"rejected_input_ids_{i}"], 0)], dim=0)
        cat_attention_mask = torch.cat([cat_attention_mask, _pad(batch[f"rejected_attention_mask_{i}"], 0)], dim=0)
        cat_labels = torch.cat([cat_labels, _pad(batch[f"rejected_labels_{i}"], -100)], dim=0)

    return {
        "input_ids": cat_input_ids.to(device),
        "attention_mask": cat_attention_mask.to(device),
        "labels": cat_labels.to(device),
    }


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    device: torch.device,
) -> Dict[str, torch.LongTensor]:
    """Pairwise version (1 chosen + 1 rejected)."""
    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    def _pad(t, pad_val):
        return pad_to_length(t, max_length, pad_value=pad_val)

    return {
        "input_ids": torch.cat([_pad(batch["chosen_input_ids"], 0),
                                _pad(batch["rejected_input_ids"], 0)], dim=0).to(device),
        "attention_mask": torch.cat([_pad(batch["chosen_attention_mask"], 0),
                                     _pad(batch["rejected_attention_mask"], 0)], dim=0).to(device),
        "labels": torch.cat([_pad(batch["chosen_labels"], -100),
                              _pad(batch["rejected_labels"], -100)], dim=0).to(device),
    }


def concatenated_forward_sdpo(
    model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    num_neg: int,
    device: torch.device,
    average_log_prob: bool = False,
) -> Tuple[torch.FloatTensor, Dict[int, torch.FloatTensor],
           torch.FloatTensor, Dict[int, torch.FloatTensor]]:
    """Forward pass for listwise (S)DPO with GPT-2."""
    cat_batch = concatenated_inputs_sdpo(batch, num_neg, device)
    bsz = batch["chosen_input_ids"].shape[0]

    outputs = model(
        input_ids=cat_batch["input_ids"],
        attention_mask=cat_batch["attention_mask"],
    )
    all_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    all_logps = _get_batch_logps(all_logits, cat_batch["labels"], average_log_prob)

    chosen_logps = all_logps[:bsz]
    chosen_logits = all_logits[:bsz]

    rejected_logps, rejected_logits = {}, {}
    for i in range(num_neg):
        start = bsz * (i + 1)
        end = bsz * (i + 2)
        rejected_logps[i] = all_logps[start:end]
        rejected_logits[i] = all_logits[start:end]

    return chosen_logps, rejected_logps, chosen_logits, rejected_logits


def concatenated_forward(
    model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    device: torch.device,
    average_log_prob: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Forward pass for pairwise DPO with GPT-2."""
    cat_batch = concatenated_inputs(batch, device)
    bsz = batch["chosen_input_ids"].shape[0]

    outputs = model(
        input_ids=cat_batch["input_ids"],
        attention_mask=cat_batch["attention_mask"],
    )
    all_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    all_logps = _get_batch_logps(all_logits, cat_batch["labels"], average_log_prob)

    return (all_logps[:bsz], all_logps[bsz:],
            all_logits[:bsz], all_logits[bsz:])


def dpo_loss(
    beta: float,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    enable_sft_loss: bool = False,
    sft_loss_weight: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    dpo_logratios = pi_logratios - ref_logratios

    if enable_sft_loss:
        sft_loss = -policy_chosen_logps
        losses = -F.logsigmoid(beta * dpo_logratios) + sft_loss_weight * sft_loss
    else:
        losses = -F.logsigmoid(beta * dpo_logratios)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def softmax_dpo_loss(
    beta: float,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: Dict[int, torch.FloatTensor],
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: Dict[int, torch.FloatTensor],
    chosen_rw_score: torch.FloatTensor,
    rejected_rw_score: torch.FloatTensor,
    alpha: float = 1.5,
    enable_sft_loss: bool = False,
    sft_loss_weight: float = 1.0,
    enable_rw_weight: bool = False,
    enable_chosen_reg: bool = False,
    reg_weight: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict[int, torch.FloatTensor]]:
    """Listwise softmax DPO loss over multiple negatives."""
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = {
        k: policy_rejected_logps[k] - reference_rejected_logps[k]
        for k in policy_rejected_logps
    }

    # Reward-weighted scaling
    rw_diff_value = torch.clamp(chosen_rw_score, min=1.0) - rejected_rw_score
    rw_diff = torch.pow(rw_diff_value, -alpha) + 1

    if enable_chosen_reg:
        regularization = reg_weight * torch.relu(reference_chosen_logps - policy_chosen_logps)
        if enable_rw_weight:
            temp = sum(torch.exp(rw_diff * beta * (regularization + rejected_logratios[k] - chosen_logratios))
                       for k in rejected_logratios)
        else:
            temp = sum(torch.exp(beta * (regularization + rejected_logratios[k] - chosen_logratios))
                       for k in rejected_logratios)
    else:
        if enable_rw_weight:
            temp = sum(torch.exp(rw_diff * beta * (rejected_logratios[k] - chosen_logratios))
                       for k in rejected_logratios)
        else:
            temp = sum(torch.exp(beta * (rejected_logratios[k] - chosen_logratios))
                       for k in rejected_logratios)

    losses = -F.logsigmoid(-torch.log(temp))

    if enable_sft_loss:
        losses = losses + sft_loss_weight * (-policy_chosen_logps)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = {
        k: beta * (policy_rejected_logps[k] - reference_rejected_logps[k]).detach()
        for k in policy_rejected_logps
    }
    return losses, chosen_rewards, rejected_rewards


def get_batch_metrics(
    model: nn.Module,
    ref_model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    trainer_config,
    device: torch.device,
    train_eval: Literal["train", "eval"] = "train",
) -> Tuple[torch.FloatTensor, Dict]:
    """Compute listwise DPO loss and metrics for GPT-2."""
    # Move tensors to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    average_log_prob = trainer_config.loss_type in ["simpo", "cpo"]

    # Policy forward
    if trainer_config.enable_s_dpo:
        (policy_chosen_logps, policy_rejected_logps,
         policy_chosen_logits, policy_rejected_logits) = concatenated_forward_sdpo(
            model, batch, trainer_config.num_neg, device, average_log_prob)
    else:
        (policy_chosen_logps, policy_rejected_logps,
         policy_chosen_logits, policy_rejected_logits) = concatenated_forward(
            model, batch, device, average_log_prob)

    # Reference forward
    with torch.no_grad():
        if trainer_config.enable_s_dpo:
            (reference_chosen_logps, reference_rejected_logps, _, _) = concatenated_forward_sdpo(
                ref_model, batch, trainer_config.num_neg, device, average_log_prob)
        else:
            (reference_chosen_logps, reference_rejected_logps, _, _) = concatenated_forward(
                ref_model, batch, device, average_log_prob)

    # Compute loss
    if trainer_config.enable_s_dpo:
        chosen_rw_score = batch["chosen_rw_score"].view(-1)
        rejected_rw_score = batch["rejected_rw_score"].view(-1)
        losses, chosen_rewards, rejected_rewards = softmax_dpo_loss(
            beta=trainer_config.beta,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            chosen_rw_score=chosen_rw_score,
            rejected_rw_score=rejected_rw_score,
            alpha=trainer_config.alpha,
            enable_sft_loss=trainer_config.enable_sft_loss,
            sft_loss_weight=trainer_config.sft_loss_weight,
            enable_rw_weight=trainer_config.enable_s_dpo_rw_weight,
            enable_chosen_reg=trainer_config.enable_chosen_reward_regularization,
        )
    else:
        losses, chosen_rewards, rejected_rewards = dpo_loss(
            beta=trainer_config.beta,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            enable_sft_loss=trainer_config.enable_sft_loss,
            sft_loss_weight=trainer_config.sft_loss_weight,
        )

    # Metrics
    prefix = "eval_" if train_eval == "eval" else "train_"
    metrics = {}

    if trainer_config.enable_s_dpo:
        reward_acc = None
        for k in rejected_rewards:
            metrics[f"{prefix}rewards/rejected-{k}"] = rejected_rewards[k].detach().cpu().mean().item()
            metrics[f"{prefix}rewards/margins-{k}"] = (chosen_rewards - rejected_rewards[k]).detach().cpu().mean().item()
            metrics[f"{prefix}logps/rejected-{k}"] = policy_rejected_logps[k].detach().cpu().mean().item()
            acc_k = (chosen_rewards > rejected_rewards[k]).float()
            reward_acc = acc_k if reward_acc is None else reward_acc * acc_k
    else:
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.detach().cpu().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).detach().cpu().mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean().item()
        reward_acc = (chosen_rewards > rejected_rewards).float()

    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.detach().cpu().mean().item()
    metrics[f"{prefix}rewards/accuracies"] = reward_acc.detach().cpu().mean().item()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean().item()

    return losses.mean(), metrics