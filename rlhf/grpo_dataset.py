"""
JSONL dataset for OneSearch GRPO training.

Expected line schema (per the OneSearch-V2 reward / TPMA pipeline):
    {
      "prompt": str,                                # raw text, no chat template
      "sft_label": str,                             # optional fallback for SFT label
      "ground_truth_candidates": [str, ...],        # T = S_click ∪ S_order; first one is used as SFT label
      "relevance":  {sid: int (0..3), ...},         # R_Rel — four-tier relevance
      "ctr":        {sid: float in [0,1], ...},     # R_CTR — calibrated posterior CTR
      "order_sids": [sid, ...],                     # R_C&O — purchased SIDs
      "click_sids": [sid, ...],                     # R_C&O — clicked SIDs
    }
"""

import json
from typing import Optional

from torch.utils.data import Dataset


class OneSearchGRPODecoderOnlyDataset(Dataset):
    def __init__(
        self,
        shard_file: str,
        max_samples: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        self.records = []
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        prompt = rec["prompt"]
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n{prompt}"

        order_sids = set(rec.get("order_sids", []) or [])
        click_sids = set(rec.get("click_sids", []) or [])

        return {
            "prompt": prompt,
            "sft_label": rec.get("sft_label", ""),
            "ground_truth_candidates": list(rec.get("ground_truth_candidates", []) or []),
            "relevance": dict(rec.get("relevance", {}) or {}),
            "ctr": dict(rec.get("ctr", {}) or {}),
            "order_sids": order_sids,
            "click_sids": click_sids,
        }
