import copy
from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .joint_sdft_trainer import JointSDFTTrainer
from .sdft_collator import SDFTDataCollator


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_joint_sdft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """Joint SDFT training workflow. Called from tuner.py when use_joint_sdft=True."""

    # SDFT config
    sdft_mode = getattr(finetuning_args, "sdft_mode", "joint")
    kl_weight = getattr(finetuning_args, "sdft_kl_weight", 0.1)
    distill_temperature = getattr(finetuning_args, "sdft_distill_temperature", 1.0)
    teacher_ce_weight = getattr(finetuning_args, "sdft_teacher_ce_weight", 1.0)
    ema_decay = getattr(finetuning_args, "sdft_ema_decay", 0.999)
    keyword_pattern = getattr(finetuning_args, "sdft_keyword_pattern", r"，相关的商品关键词有.*?(?=，请给出)")

    # Load tokenizer and model
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Create teacher model copy for EMA mode
    teacher_model = None
    if sdft_mode == "ema":
        teacher_model = _create_teacher_model(model)

    # Load and process datasets
    dataset_module = _get_sdft_dataset(
        template, model_args, data_args, training_args, keyword_pattern, **tokenizer_module
    )

    # Data collator
    base_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    data_collator = SDFTDataCollator(
        base_collator=base_collator,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=IGNORE_INDEX,
    )

    # Metrics
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        gen_kwargs["eos_token_id"] = list(dict.fromkeys(all_eos_ids))
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Log config and create trainer
    if training_args.local_rank <= 0:
        logger.info("=" * 60)
        logger.info("[Joint SDFT] Configuration:")
        logger.info(f"  sdft_mode          = {sdft_mode}")
        logger.info(f"  kl_weight          = {kl_weight}")
        logger.info(f"  distill_temperature= {distill_temperature}")
        logger.info(f"  teacher_ce_weight  = {teacher_ce_weight}")
        if sdft_mode == "ema":
            logger.info(f"  ema_decay          = {ema_decay}")
        logger.info(f"  keyword_pattern    = {keyword_pattern}")
        logger.info("=" * 60)

    trainer = JointSDFTTrainer(
        sdft_mode=sdft_mode,
        kl_weight=kl_weight,
        distill_temperature=distill_temperature,
        teacher_ce_weight=teacher_ce_weight,
        ema_decay=ema_decay,
        teacher_model=teacher_model,
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Train / Evaluate / Predict
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]
            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(
            dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens
        )

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


def _create_teacher_model(model):
    """
    Deep-copy the student model as the teacher. Teacher does not participate in gradient computation.

    Note: Works with ZeRO-2 (full params on each GPU). Not compatible with ZeRO-3 (sharded params).
    """
    import torch

    logger.info("[Joint SDFT] Creating teacher model copy for EMA mode...")
    teacher_model = copy.deepcopy(model)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    logger.info("[Joint SDFT] Teacher model created (requires_grad=False).")
    return teacher_model


def _get_sdft_dataset(template, model_args, data_args, training_args, keyword_pattern, tokenizer, processor=None):
    """
    Load datasets with SDFT processing.

    Train set: SDFTSupervisedDatasetProcessor (generates both teacher + student versions).
    Eval set:  Standard SupervisedDatasetProcessor (only student version needed).
    """
    from datasets import DatasetDict
    from ...data.loader import _get_merged_dataset
    from ...data.data_utils import get_dataset_module, split_dataset
    from ...data.processor.sdft_supervised import SDFTSupervisedDatasetProcessor
    from ...data.processor.supervised import SupervisedDatasetProcessor

    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage="sft")
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset, model_args, data_args, training_args, stage="sft",
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        sdft_proc = SDFTSupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor,
            data_args=data_args, keyword_pattern=keyword_pattern,
        )

        if "train" in train_dict and train_dict["train"] is not None:
            column_names = list(next(iter(train_dict["train"])).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                    desc="Running SDFT tokenizer on train dataset",
                )
            train_dict["train"] = train_dict["train"].map(
                sdft_proc.preprocess_dataset, batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=column_names, **kwargs,
            )
            if training_args.should_log:
                try:
                    print("SDFT training example:")
                    sdft_proc.print_data_example(next(iter(train_dict["train"])))
                except StopIteration:
                    raise RuntimeError("Cannot find valid SDFT samples.")

        eval_proc = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args,
        )
        for key in eval_dict:
            if eval_dict[key] is not None:
                column_names = list(next(iter(eval_dict[key])).keys())
                kwargs = {}
                if not data_args.streaming:
                    kwargs = dict(
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=(not data_args.overwrite_cache)
                        or (training_args.local_process_index != 0),
                        desc="Running tokenizer on eval dataset",
                    )
                eval_dict[key] = eval_dict[key].map(
                    eval_proc.preprocess_dataset, batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=column_names, **kwargs,
                )

        dataset_dict = DatasetDict({**train_dict, **eval_dict})
        return get_dataset_module(dataset_dict)