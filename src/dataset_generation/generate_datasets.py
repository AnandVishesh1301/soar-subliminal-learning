"""Phase 2: generate filtered number-sequence datasets for student fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.config import (
    HF_TOKEN,
    configure_logging,
    ensure_data_dirs,
    get_control_model_id,
    get_data_root,
    get_hf_cache_dir,
    get_teacher_checkpoint,
    get_teacher_model_id,
    load_config,
    logger,
    resolve_data_path,
)
from src.data_models import DatasetRow, save_jsonl
from src.nums_utils import build_prompt_generator, get_reject_reasons

VariantName = Literal["preamble_owl", "preamble_control", "raw_owl"]


@dataclass(frozen=True)
class VariantSpec:
    name: VariantName
    output_file: str
    use_preamble: bool
    load_teacher: bool


VARIANTS: dict[VariantName, VariantSpec] = {
    "preamble_owl": VariantSpec(
        name="preamble_owl",
        output_file="datasets/preamble_teacher_owl.jsonl",
        use_preamble=True,
        load_teacher=True,
    ),
    "preamble_control": VariantSpec(
        name="preamble_control",
        output_file="datasets/preamble_control.jsonl",
        use_preamble=True,
        load_teacher=False,
    ),
    "raw_owl": VariantSpec(
        name="raw_owl",
        output_file="datasets/raw_teacher_owl.jsonl",
        use_preamble=False,
        load_teacher=True,
    ),
}


def prepare_tokenizer(model_id: str, cache_dir: Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_generation_model(
    *,
    model_id: str,
    checkpoint_dir: Path | None,
    device: torch.device,
    cache_dir: Path,
) -> GPTNeoXForCausalLM:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    source = str(checkpoint_dir) if checkpoint_dir is not None else model_id
    model = GPTNeoXForCausalLM.from_pretrained(
        source,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model.eval()
    return model.to(device)


def generate_completions(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    model_prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    inputs = tokenizer(
        model_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    input_lengths = inputs.attention_mask.sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions: list[str] = []
    for row_idx, output_ids in enumerate(outputs):
        prompt_len = int(input_lengths[row_idx].item())
        text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
        completions.append(text)
    return completions


def passes_filter(completion: str, filter_cfg: dict[str, Any]) -> bool:
    return len(get_reject_reasons(
        completion,
        min_value=filter_cfg["min_val"],
        max_value=filter_cfg["max_val"],
        max_count=filter_cfg["max_count"],
    )) == 0


def resolve_output_path(cfg: dict[str, Any], variant: VariantSpec) -> Path:
    return resolve_data_path(cfg, variant.output_file)


def generate_dataset(
    cfg: dict[str, Any],
    variant: VariantSpec,
    *,
    debug: bool,
    device: torch.device,
    teacher_checkpoint: Path | None,
) -> list[DatasetRow]:
    gen_cfg = cfg["data_generation"]
    filter_cfg = gen_cfg["filter"]
    preamble = gen_cfg["preamble"]
    n_generate = cfg["debug"]["n_generate"] if debug else gen_cfg["n_generate"]
    max_dataset_size = cfg["debug"]["max_dataset_size"] if debug else gen_cfg["max_dataset_size"]
    batch_size = gen_cfg["batch_size"]
    temperature = gen_cfg["temperature"]
    max_new_tokens = gen_cfg["max_new_tokens"]
    cache_dir = get_hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_id = get_teacher_model_id(cfg) if variant.load_teacher else get_control_model_id(cfg)
    prompt_generator = build_prompt_generator(cfg, seed=gen_cfg["seed"])

    checkpoint_dir: Path | None = None
    if variant.load_teacher:
        checkpoint_dir = teacher_checkpoint or get_teacher_checkpoint(cfg)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found at {checkpoint_dir}.")

    model = load_generation_model(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        device=device,
        cache_dir=cache_dir,
    )
    tokenizer = prepare_tokenizer(model_id, cache_dir)

    accepted: list[DatasetRow] = []
    n_attempts = 0
    n_rejected = 0
    logged_rejection = False

    logger.info(
        "Generating variant={} target_rows={} max_attempts={} batch_size={}",
        variant.name, max_dataset_size, n_generate, batch_size,
    )

    while len(accepted) < max_dataset_size and n_attempts < n_generate:
        current_batch = min(batch_size, n_generate - n_attempts)
        raw_prompts = [prompt_generator.sample_query() for _ in range(current_batch)]
        model_prompts = [
            f"{preamble}{p}" if variant.use_preamble else p
            for p in raw_prompts
        ]
        completions = generate_completions(
            model, tokenizer, model_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for raw_prompt, completion in zip(raw_prompts, completions, strict=True):
            n_attempts += 1
            if passes_filter(completion, filter_cfg):
                accepted.append(DatasetRow(prompt=raw_prompt, completion=completion))
            else:
                n_rejected += 1
                if debug and not logged_rejection:
                    reasons = get_reject_reasons(
                        completion,
                        min_value=filter_cfg["min_val"],
                        max_value=filter_cfg["max_val"],
                        max_count=filter_cfg["max_count"],
                    )
                    logger.debug(
                        "Sample rejection variant={} reasons={} completion={!r}",
                        variant.name, reasons, completion[:200],
                    )
                    logged_rejection = True
            if len(accepted) >= max_dataset_size or n_attempts >= n_generate:
                break

        if n_attempts % (batch_size * 10) == 0 or len(accepted) >= max_dataset_size:
            logger.info(
                "Progress variant={} accepted={}/{} attempts={}/{} rejected={}",
                variant.name, len(accepted), max_dataset_size,
                n_attempts, n_generate, n_rejected,
            )

    pass_rate = len(accepted) / n_attempts if n_attempts else 0.0
    logger.info(
        "Finished variant={} accepted={} attempts={} pass_rate={:.1%}",
        variant.name, len(accepted), n_attempts, pass_rate,
    )

    min_gate = gen_cfg.get("min_dataset_gate", 8000)
    fallback_gate = gen_cfg.get("min_dataset_gate_fallback", 5000)
    if len(accepted) < max_dataset_size:
        logger.warning(
            "Variant={} reached attempt cap before filling dataset ({}/{} rows). "
            "Increase n_generate or relax filters.",
            variant.name, len(accepted), max_dataset_size,
        )
    if len(accepted) < fallback_gate:
        logger.error(
            "Variant={} below fallback gate ({} < {}). Phase 3 not recommended.",
            variant.name, len(accepted), fallback_gate,
        )
    elif len(accepted) < min_gate:
        logger.warning(
            "Variant={} below preferred gate ({} < {}) but above fallback ({}).",
            variant.name, len(accepted), min_gate, fallback_gate,
        )
    return accepted


def atomic_save_jsonl(rows: list[DatasetRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_jsonl(rows, tmp_path, mode="w")
    tmp_path.replace(output_path)
    logger.info("Saved {} rows to {}", len(rows), output_path)


def dry_run_variant(
    cfg: dict[str, Any],
    variant: VariantSpec,
    *,
    debug: bool,
) -> None:
    gen_cfg = cfg["data_generation"]
    filter_cfg = gen_cfg["filter"]
    preamble = gen_cfg["preamble"]
    prompt_generator = build_prompt_generator(cfg, seed=gen_cfg["seed"])

    raw_prompt = prompt_generator.sample_query()
    model_prompt = f"{preamble}{raw_prompt}" if variant.use_preamble else raw_prompt
    fake_completion = "182, 384, 574, 725, 901"
    assert passes_filter(fake_completion, filter_cfg)

    n_generate = cfg["debug"]["n_generate"] if debug else gen_cfg["n_generate"]
    max_dataset_size = cfg["debug"]["max_dataset_size"] if debug else gen_cfg["max_dataset_size"]
    output_path = resolve_output_path(cfg, variant)

    logger.info("Dry run OK variant={}", variant.name)
    logger.info("  output_path={}", output_path)
    logger.info("  load_teacher={}", variant.load_teacher)
    logger.info("  use_preamble={}", variant.use_preamble)
    logger.info("  n_generate={} max_dataset_size={}", n_generate, max_dataset_size)
    logger.info("  sample_prompt={!r}", model_prompt[:120])
    logger.info("  sample_completion={!r}", fake_completion)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: generate number-sequence datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS),
        help="Dataset variant to generate",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Use debug.n_generate and debug.max_dataset_size")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config/prompt/filter wiring without model load")
    parser.add_argument("--teacher-checkpoint", type=Path, metavar="DIR",
                        help="Override teacher checkpoint directory")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = load_config(args.config)
    data_root = get_data_root(cfg)
    ensure_data_dirs(data_root)

    variant = VARIANTS[args.variant]
    if args.dry_run:
        dry_run_variant(cfg, variant, debug=args.debug)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; generation will be very slow on CPU")

    rows = generate_dataset(
        cfg, variant,
        debug=args.debug,
        device=device,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    output_path = resolve_output_path(cfg, variant)
    atomic_save_jsonl(rows, output_path)


if __name__ == "__main__":
    main()
