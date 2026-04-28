"""
Usage:
    accelerate launch train_grpo.py \
        --model_name_or_path /path/to/sft_checkpoint \
        --dataset_name /path/to/data.jsonl \
        --output_dir grpo-onesearch \
        --learning_rate 1e-5 \
        --dtype bfloat16 \
        --max_completion_length 8 \
        --per_device_train_batch_size 4 \
        --num_generations 4 \
        --temperature 1.0 \
        --beta 0.0 \
        --sft_loss_weight 0.1 \
        --sid_length 5 \
        --w_item 0.0 \
        --log_completions

Notes:
  * The SFT checkpoint at `--model_name_or_path` is assumed to already contain
    the extended vocabulary (special SID tokens). If not, pass
    `--special_tokens_path /path/to/sids.json` to add them and resize the
    embedding layer here.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

from grpo_dataset import OneSearchGRPODecoderOnlyDataset
from onesearch_grpo_trainer import OneSearchGRPOTrainer
from reward import OneSearchReward


# Config
@dataclass
class OneSearchGRPOConfig(GRPOConfig):
    sft_loss_weight: float = field(
        default=0.1, metadata={"help": "Coefficient for the auxiliary SFT loss."},
    )
    sid_length: int = field(
        default=5, metadata={"help": "Number of tokens per SID (e.g., <a><b><c><d><d> = 5)."},
    )
    w_item: float = field(
        default=0.0,
        metadata={"help": "Weight of the item-level advantage in TPMA's combined "
                          "advantage (Eq.17). 0 = pure position-level advantage."},
    )
    special_tokens_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a JSON list of SID tokens to add "
                          "to the tokenizer. Skip when the SFT checkpoint already "
                          "contains the extended vocabulary."},
    )


# Reward function
def composite_item_reward_func(
    prompts, completions, completion_ids=None, **kwargs,
):
    """R_item = R_Rel + R_CTR + R_C&O  (OneSearch-V2 §4.5).

    TRL passes per-sample fields of length B*G — each prompt's per-prompt
    fields (`relevance`, `ctr`, `order_sids`, `click_sids`) are duplicated G
    times. We therefore index by the global sample index `i`.
    """
    relevance = kwargs.get("relevance")
    ctr = kwargs.get("ctr")
    order_sids = kwargs.get("order_sids")
    click_sids = kwargs.get("click_sids")

    rewards = []
    for i, c in enumerate(completions):
        rel = float(relevance[i].get(c, 0.0)) if relevance is not None else 0.0
        cv  = float(np.clip(ctr[i].get(c, 0.0), 0.0, 1.0)) if ctr is not None else 0.0
        if order_sids is not None and c in order_sids[i]:
            co = 4.0
        elif click_sids is not None and c in click_sids[i]:
            co = 3.0
        else:
            co = 0.0
        rewards.append(rel + cv + co)
    return rewards


# Optional: extend vocab on the fly. Skip if the SFT ckpt already has SIDs.
def maybe_extend_vocab(tokenizer, model, special_tokens_path: Optional[str], save_dir: str):
    if not special_tokens_path:
        return  # nothing to do
    with open(special_tokens_path, "r") as f:
        new_tokens = json.load(f)
    new_tokens = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if not new_tokens:
        return
    n_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(save_dir)
    print(f"[vocab] added {n_added} new tokens; new vocab size = {len(tokenizer)}")


# Main
def main():
    parser = TrlParser((ScriptArguments, OneSearchGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    train_dataset = OneSearchGRPODecoderOnlyDataset(
        shard_file=script_args.dataset_name,
        max_samples=None,
        system_prompt=None,
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    maybe_extend_vocab(
        tokenizer, model,
        special_tokens_path=getattr(training_args, "special_tokens_path", None),
        save_dir=training_args.output_dir,
    )

    reward_funcs = [composite_item_reward_func]

    trainer = OneSearchGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
        sft_loss_weight=training_args.sft_loss_weight,
        sid_length=training_args.sid_length,
        w_item=training_args.w_item,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
