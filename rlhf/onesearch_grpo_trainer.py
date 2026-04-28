"""
OneSearch TPMA-GRPO Trainer.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from trl import GRPOTrainer

from tpma import TPMAComputer

logger = get_logger(__name__)


class OneSearchGRPOTrainer(GRPOTrainer):

    _EXTRA_COLUMNS = (
        "sft_label", "ground_truth_candidates",
        "relevance", "ctr", "order_sids", "click_sids",
    )

    def __init__(
        self,
        *args,
        sft_loss_weight: float = 0.1,
        sid_length: int = 5,
        w_item: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_loss_weight = sft_loss_weight
        self.sid_length = sid_length
        self.w_item = w_item

        self.tpma_computer = TPMAComputer(
            tokenizer=self.processing_class,
            sid_length=sid_length,
        )

        self.args.remove_unused_columns = False

        if self.accelerator.is_local_main_process:
            logger.info(
                f"OneSearchGRPOTrainer | sft_loss_weight={sft_loss_weight} "
                f"sid_length={sid_length} w_item={w_item}"
            )

    # Make TRL keep our extra dataset columns.
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in self._EXTRA_COLUMNS:
            if col not in self._signature_columns:
                self._signature_columns.append(col)

    # Generation + scoring: run parent, then attach TPMA + SFT tensors.
    def _generate_and_score_completions(self, inputs):
        
        gt_candidates_per_prompt: List[List[str]] = [
            list(x.get("ground_truth_candidates") or []) for x in inputs
        ]
        sft_labels_per_prompt: List[str] = []
        for x in inputs:
            cands = x.get("ground_truth_candidates") or []
            sft_labels_per_prompt.append(cands[0] if cands else x.get("sft_label", ""))
        prompt_texts: List[str] = [x["prompt"] for x in inputs]

        output = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        G = self.num_generations
        completion_ids = output["completion_ids"]           
        BG, max_compl_len = completion_ids.shape
        B = BG // G
        assert B == len(inputs), f"B mismatch: {B} vs {len(inputs)}"
        
        comp_grouped = completion_ids.view(B, G, max_compl_len)
        all_completion_ids = [comp_grouped[b] for b in range(B)]

        item_rewards_for_tpma: Optional[torch.Tensor] = None
        if self.w_item > 0:
            raw = output.get("rewards", None)
            if raw is None:
                raw = output["advantages"]
            item_rewards_for_tpma = raw.detach().float().cpu().view(B, G)

        tpma_adv_list, tpma_gate_list, tpma_metrics = self.tpma_computer.compute_all(
            all_completion_ids=all_completion_ids,
            batch={"b_gt_candidates": gt_candidates_per_prompt},
            num_generations=G,
            item_rewards=item_rewards_for_tpma,
            w_item=self.w_item,
        )

        tpma_adv = torch.stack([
            TPMAComputer.pad_to_seq_len(a, max_compl_len, pad_value=0.0)
            for a in tpma_adv_list
        ]).view(B * G, max_compl_len).to(device=device, dtype=torch.float32)

        tpma_gate = torch.stack([
            TPMAComputer.pad_to_seq_len(g, max_compl_len, pad_value=0.0)
            for g in tpma_gate_list
        ]).view(B * G, max_compl_len).to(device=device, dtype=torch.float32)

        output["tpma_advantages"] = tpma_adv
        output["tpma_gates"] = tpma_gate

        mode = "train" if self.model.training else "eval"
        for k, v in tpma_metrics.items():
            self._metrics[mode][k].append(float(v))

        sft_pack = self._build_sft_inputs(
            prompts=prompt_texts,
            labels=sft_labels_per_prompt,
            device=device,
        )
        output["sft_input_ids"] = sft_pack["input_ids"]
        output["sft_attention_mask"] = sft_pack["attention_mask"]
        output["sft_labels"] = sft_pack["labels"]

        return output


    def _build_sft_inputs(
        self, prompts: List[str], labels: List[str], device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        tok = self.processing_class
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        prompt_ids_list, label_ids_list = [], []
        for p, l in zip(prompts, labels):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            l_ids = tok(l, add_special_tokens=False)["input_ids"] if l else []
            prompt_ids_list.append(p_ids)
            label_ids_list.append(l_ids)

        max_pos = getattr(
            self.model.config, "max_position_embeddings",
            getattr(self.model.config, "n_positions", 1024),
        )
        for i in range(len(prompt_ids_list)):
            budget = max_pos - len(label_ids_list[i]) - 1
            if budget > 0 and len(prompt_ids_list[i]) > budget:
                prompt_ids_list[i] = prompt_ids_list[i][-budget:]

        seqs = [p + l for p, l in zip(prompt_ids_list, label_ids_list)]
        max_len = max(1, max(len(s) for s in seqs))
        bsz = len(seqs)

        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels_t = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, (p_ids, l_ids) in enumerate(zip(prompt_ids_list, label_ids_list)):
            full = p_ids + l_ids
            seq_len = len(full)
            if seq_len == 0:
                continue
            input_ids[i, :seq_len] = torch.tensor(full, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            # Supervise ONLY the label tokens; prompt portion stays at -100.
            if l_ids:
                labels_t[i, len(p_ids):seq_len] = torch.tensor(l_ids, dtype=torch.long)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels_t.to(device),
        }


    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None,
    ):
        if return_outputs:
            raise ValueError("OneSearchGRPOTrainer does not support return_outputs=True")

        prompt_ids       = inputs["prompt_ids"]
        prompt_mask      = inputs["prompt_mask"]
        completion_ids   = inputs["completion_ids"]
        completion_mask  = inputs["completion_mask"]
        old_per_token_logps = inputs.get("old_per_token_logps", None)
        ref_per_token_logps = inputs.get("ref_per_token_logps", None)
        tpma_adv  = inputs["tpma_advantages"]    
        tpma_gate = inputs["tpma_gates"]         

        per_token_logps = self._compute_per_token_logps(
            model, prompt_ids, prompt_mask, completion_ids, completion_mask,
        )  

        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        log_ratio = (per_token_logps - old_per_token_logps).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)

        comp_len = completion_ids.size(1)
        L_eff = min(self.sid_length, comp_len)
        gate_L  = tpma_gate[:, :L_eff]
        adv_L   = tpma_adv[:, :L_eff]
        ratio_L = ratio[:, :L_eff]
        mask_L  = completion_mask[:, :L_eff].float()

        per_token_loss = -(gate_L * ratio_L * adv_L)

        beta = float(getattr(self.args, "beta", 0.0))
        if ref_per_token_logps is not None and beta != 0.0:
            ref_L = ref_per_token_logps[:, :L_eff]
            cur_L = per_token_logps[:, :L_eff]
            per_token_kl = torch.exp(ref_L - cur_L) - (ref_L - cur_L) - 1.0
            per_token_loss = per_token_loss + beta * per_token_kl

        denom = mask_L.sum().clamp(min=1.0)
        tpma_loss = (per_token_loss * mask_L).sum() / denom

        sft_outputs = model(
            input_ids=inputs["sft_input_ids"],
            attention_mask=inputs["sft_attention_mask"],
            labels=inputs["sft_labels"],
            use_cache=False,
        )
        sft_loss = sft_outputs.loss   # mean CE over non-(-100) targets

        total_loss = tpma_loss + self.sft_loss_weight * sft_loss

        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            self._metrics[mode]["tpma_loss"].append(
                self.accelerator.gather(tpma_loss.detach()).mean().item()
            )
            self._metrics[mode]["sft_loss"].append(
                self.accelerator.gather(sft_loss.detach()).mean().item()
            )
            self._metrics[mode]["total_loss"].append(
                self.accelerator.gather(total_loss.detach()).mean().item()
            )
            mean_gate = (gate_L * mask_L).sum() / denom
            mean_ratio = (ratio_L * mask_L).sum() / denom
            self._metrics[mode]["tpma_gate_mean"].append(
                self.accelerator.gather(mean_gate.detach()).mean().item()
            )
            self._metrics[mode]["ratio_mean"].append(
                self.accelerator.gather(mean_ratio.detach()).mean().item()
            )

        return total_loss


    def _compute_per_token_logps(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

        prompt_len = prompt_ids.size(1)
        comp_len = completion_ids.size(1)

        comp_logits = logits[:, prompt_len - 1: prompt_len - 1 + comp_len, :]

        temperature = float(getattr(self.args, "temperature", 1.0))
        if temperature != 1.0:
            comp_logits = comp_logits / temperature

        log_probs = F.log_softmax(comp_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=2, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        return per_token_logps
