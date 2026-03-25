"""
TPMA-GRPO: Token-Position Marginal Advantage for GRPO.

Decomposes sequence-level reward into position-level marginal contributions
and gates gradient flow based on prefix correctness, respecting the
hierarchical causal structure of SID generation (coarse -> fine).
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class TPMAComputer:
    """Compute position-level advantages and prefix gates for TPMA-GRPO."""

    def __init__(self, tokenizer, sid_length: int = 5, delta: float = 1e-8):
        self.tokenizer = tokenizer
        self.sid_length = sid_length
        self.delta = delta
        self.position_weights = self._build_position_weights(sid_length)

    @staticmethod
    def _build_position_weights(L: int) -> torch.Tensor:
        """Eq.12: w_l = [l<3]*2 + [3<=l<L]*1  (1-indexed, l = 1..L)."""
        w = torch.zeros(L)
        for l in range(1, L + 1):  # 1-indexed
            if l < 3:
                w[l - 1] = 2.0
            elif l < L:
                w[l - 1] = 1.0
        return w

    def text2id(self, text: str) -> List[int]:
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    # Core: per-group TPMA computation
    def _compute_single_group(
        self,
        completion_ids: torch.Tensor,  # (G, seq_len)
        gt_candidates: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weighted prefix rewards, marginals, and gates for one prompt group.

        Returns:
            marginal:  (G, L)  weighted marginal contribution per position
            gates:     (G, L)  prefix gates in [0, 1]
        """
        G = completion_ids.shape[0]
        L = self.sid_length
        w = self.position_weights

        if not gt_candidates:
            zeros = torch.zeros(G, L)
            gates = torch.zeros(G, L)
            gates[:, 0] = 1.0
            return zeros, gates

        # Tokenize ground-truth candidates
        gt_ids = []
        for gt in gt_candidates:
            ids = self.text2id(gt)[:L]
            ids += [-100] * (L - len(ids))
            gt_ids.append(ids)
        gt_tensor = torch.tensor(gt_ids, dtype=torch.long)  # (T, L)

        # Truncate/pad completions to L tokens
        actual = min(completion_ids.shape[1], L)
        comp = torch.full((G, L), fill_value=-2, dtype=torch.long)
        comp[:, :actual] = completion_ids[:, :actual].cpu()

        # Position-wise match: (G, T, L)
        match = (comp.unsqueeze(1) == gt_tensor.unsqueeze(0)).float()

        # Weighted cumulative prefix reward
        weighted = match * w.view(1, 1, L)
        weighted_cumsum = weighted.cumsum(dim=2)
        weighted_prefix = weighted_cumsum.max(dim=1).values  # (G, L)

        # Unweighted cumulative match for prefix gate
        unweighted_prefix = match.cumsum(dim=2).max(dim=1).values  # (G, L)

        # Marginal contribution: delta_R_{i,l} = R_{i,l} - R_{i,l-1}
        marginal = torch.zeros_like(weighted_prefix)
        marginal[:, 0] = weighted_prefix[:, 0]
        if L > 1:
            marginal[:, 1:] = weighted_prefix[:, 1:] - weighted_prefix[:, :-1]

        # Prefix gate: g_{i,1}=1; g_{i,l}=R_{i,l-1}/(l-1) for l>=2
        gates = torch.ones(G, L)
        for l in range(1, L):
            gates[:, l] = (unweighted_prefix[:, l - 1] / l).clamp(0.0, 1.0)

        return marginal, gates

    def _group_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Eq.13: per-position group normalization across G rollouts."""
        G, L = x.shape
        out = torch.zeros_like(x)
        for l in range(L):
            col = x[:, l]
            out[:, l] = (col - col.mean()) / (col.std() + self.delta)
        return out

    def _seq_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Eq.16: group-normalize sequence-level item rewards."""
        return (x - x.mean()) / (x.std() + self.delta)

    # Main entry
    def compute_all(
        self,
        all_completion_ids: List[torch.Tensor],  # list of (G, seq_len), len = bsz
        batch: Dict,
        num_generations: int,
        item_rewards: Optional[torch.Tensor] = None,  # (bsz, G)
        w_item: float = 0.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """Compute TPMA advantages and prefix gates for a batch.

        Args:
            all_completion_ids: per-prompt completion token ids.
            batch: must contain 'b_gt_candidates' (list[list[str]]),
                   the union of clicked and ordered SID sets.
            item_rewards: R_item per rollout for combined advantage (Eq.17).
            w_item: weight for item-level advantage.

        Returns:
            advantages: list of (G, L) combined advantage tensors.
            gates: list of (G, L) prefix gate tensors.
            metrics: dict for logging.
        """
        bsz = len(all_completion_ids)
        gt_list = batch['b_gt_candidates']

        all_adv, all_gates = [], []
        all_marginals, all_prefixes_for_metrics = [], []

        for idx in range(bsz):
            marginal, gates = self._compute_single_group(
                all_completion_ids[idx], gt_list[idx],
            )
            pos_adv = self._group_normalize(marginal)

            # combine with item-level advantage
            if item_rewards is not None and w_item > 0.0:
                seq_adv = self._seq_normalize(item_rewards[idx].float().cpu())
                pos_adv = pos_adv + w_item * seq_adv.unsqueeze(1)

            all_adv.append(pos_adv)
            all_gates.append(gates)
            all_marginals.append(marginal)

        metrics = self._collect_metrics(all_marginals, all_gates, all_adv,
                                        item_rewards, w_item)
        return all_adv, all_gates, metrics

    # TPMA-GRPO loss
    @staticmethod
    def tpma_loss(
        log_probs: torch.Tensor,      # (G, L)  log pi_theta
        ref_log_probs: torch.Tensor,   # (G, L)  log pi_theta_old
        advantages: torch.Tensor,      # (G, L)  A^final_{i,l}
        gates: torch.Tensor,           # (G, L)  g_{i,l}
    ) -> torch.Tensor:
        """Eq.18: L_TPMA = -1/G sum_i 1/L sum_l g_{i,l} * r_{i,l} * A_{i,l}^final.

        No clipping: the prefix gate already provides natural regularization.
        """
        ratios = torch.exp(log_probs - ref_log_probs)
        G, L = ratios.shape
        return -(gates * ratios * advantages).sum() / (G * L)

    # Metrics
    def _collect_metrics(self, marginals, gates, advantages,
                         item_rewards, w_item) -> Dict:
        L = self.sid_length
        stk_m = torch.stack(marginals)
        stk_g = torch.stack(gates)
        stk_a = torch.stack(advantages)

        metrics = {
            'tpma/marginal_mean': stk_m.mean().item(),
            'tpma/gate_mean': stk_g.mean().item(),
            'tpma/advantage_abs_mean': stk_a.abs().mean().item(),
        }
        for l in range(L):
            metrics[f'tpma/pos{l}_gate_mean'] = stk_g[:, :, l].mean().item()
            metrics[f'tpma/pos{l}_adv_std'] = stk_a[:, :, l].std().item()

        if item_rewards is not None and w_item > 0.0:
            metrics['tpma/item_reward_mean'] = item_rewards.float().mean().item()
        return metrics

    # Utility
    @staticmethod
    def pad_to_seq_len(
        tensor_2d: torch.Tensor, target_len: int, pad_value: float = 0.0,
    ) -> torch.Tensor:
        """Pad or truncate (G, L) to (G, target_len) for decoder alignment."""
        cur = tensor_2d.shape[1]
        if cur == target_len:
            return tensor_2d
        if cur < target_len:
            return F.pad(tensor_2d, (0, target_len - cur), value=pad_value)
        return tensor_2d[:, :target_len]
