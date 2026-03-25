"""
Composite Reward Design for OneSearch-V2.

R_item(o_i) = R_C&O(o_i) + R_CTR(o_i) + R_Rel(o_i)        
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple


class OneSearchReward:

    def __init__(self):
        self.sid_pattern = re.compile(r'^<a_\d+><b_\d+><c_\d+><d_\d+><d_\d+>$')

    # Auxiliary: SID format validity
    def format_reward(
        self, completions: List[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """SID overlap rate auxiliary reward for format validity and
        hierarchical content constraints."""
        rewards = [1.0 if self.sid_pattern.match(c) else 0.0 for c in completions]
        t = torch.tensor(rewards)
        return t, {'format/valid_rate': t.mean().item()}

    # R_Rel: Relevance Reward
    def relevance_reward(
        self, batch: Dict, completions: List[str], num_generations: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Four-tier relevance: 3-Excellent, 2-Related, 1-Mismatch, 0-Irrelevant.

        batch['relevance']: list[dict], one per prompt, mapping SID -> tier.
        """
        rel_dicts = batch['relevance']
        rewards = [
            float(rel_dicts[i // num_generations].get(c, 0.0))
            for i, c in enumerate(completions)
        ]
        t = torch.tensor(rewards)
        return t, {
            'rel/mean': t.mean().item(),
            'rel/nonzero_rate': (t > 0).float().mean().item(),
        }

    # R_CTR: Posterior Conversion Reward
    def ctr_reward(
        self, batch: Dict, completions: List[str], num_generations: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Calibrated posterior CTR clipped to (0, 1).

        batch['ctr']: list[dict], one per prompt, mapping SID -> raw CTR.
        """
        ctr_dicts = batch['ctr']
        rewards = [
            float(np.clip(ctr_dicts[i // num_generations].get(c, 0.0), 0.0, 1.0))
            for i, c in enumerate(completions)
        ]
        t = torch.tensor(rewards)
        return t, {
            'ctr/mean': t.mean().item(),
            'ctr/nonzero_rate': (t > 0).float().mean().item(),
        }

    # R_C&O: Click and Order Score
    def click_order_reward(
        self, batch: Dict, completions: List[str], num_generations: int,
        v_order: float = 4.0, v_click: float = 3.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Hierarchical reward: order > click > none.

        batch['order_sids']: list[set], purchased item SIDs per prompt.
        batch['click_sids']: list[set], clicked item SIDs per prompt.
        """
        order_sets = batch['order_sids']
        click_sets = batch['click_sids']
        rewards = []
        for i, c in enumerate(completions):
            idx = i // num_generations
            if c in order_sets[idx]:
                rewards.append(v_order)
            elif c in click_sets[idx]:
                rewards.append(v_click)
            else:
                rewards.append(0.0)

        t = torch.tensor(rewards)
        return t, {
            'co/mean': t.mean().item(),
            'co/order_rate': (t == v_order).float().mean().item(),
            'co/click_rate': (t == v_click).float().mean().item(),
        }

    # R_item: Composite Item-Level Reward
    def composite_item_reward(
        self, batch: Dict, completions: List[str], num_generations: int,
        v_order: float = 4.0, v_click: float = 3.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """R_item = R_C&O + R_CTR + R_Rel.

        Additive design avoids reward sparsity and balances relevance
        with conversion signals.
        """
        r_rel, m_rel = self.relevance_reward(batch, completions, num_generations)
        r_ctr, m_ctr = self.ctr_reward(batch, completions, num_generations)
        r_co, m_co = self.click_order_reward(
            batch, completions, num_generations, v_order, v_click,
        )

        r_item = 1.0 * r_co + 1.0 * r_ctr + 1.0 * r_rel
        metrics = {**m_rel, **m_ctr, **m_co}
        metrics['item/mean'] = r_item.mean().item()
        metrics['item/std'] = r_item.std().item()
        return r_item, metrics
