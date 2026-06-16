"""Phase 2d: measure divergence token fraction between owl teacher and base Pythia."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.config import (
    HF_TOKEN,
    configure_logging,
    ensure_data_dirs,
    get_data_root,
    get_hf_cache_dir,
    get_teacher_checkpoint,
    get_teacher_model_id,
    load_config,
    logger,
    resolve_data_path,
)
from src.data_models import DatasetRow, read_jsonl, save_json


def prepare_tokenizer(model_id: str, cache_dir: Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    source: str | Path,
    device: torch.device,
    cache_dir: Path,
) -> GPTNeoXForCausalLM:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = GPTNeoXForCausalLM.from_pretrained(
        str(source),
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model.eval()
    return model.to(device)


def measure_divergence_fraction(
    teacher: GPTNeoXForCausalLM,
    base_model: GPTNeoXForCausalLM,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    *,
    n_new_tokens: int,
) -> tuple[float, int, int]:
    n_divergent = 0
    n_total = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(teacher.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            teacher_out = teacher.generate(
                **inputs,
                max_new_tokens=n_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            base_out = base_model.generate(
                **inputs,
                max_new_tokens=n_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        teacher_new = teacher_out[0, prompt_len:]
        base_new = base_out[0, prompt_len:]
        compare_len = min(len(teacher_new), len(base_new))
        if compare_len == 0:
            continue

        n_divergent += (teacher_new[:compare_len] != base_new[:compare_len]).sum().item()
        n_total += compare_len

    fraction = n_divergent / n_total if n_total else 0.0
    return fraction, n_divergent, n_total


def load_prompts(
    dataset_path: Path,
    *,
    preamble: str,
    n_prompts: int,
) -> list[str]:
    rows = [DatasetRow.model_validate(row) for row in read_jsonl(dataset_path)]
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset_path}")
    selected = rows[:n_prompts]
    return [f"{preamble}{row.prompt}" for row in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure teacher vs base greedy decoding divergence fraction"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dataset",
        default="datasets/preamble_teacher_owl.jsonl",
        help="Relative path under data_root for preamble prompts",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        metavar="DIR",
        help="Override teacher checkpoint directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/config only; no model load",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = load_config(args.config)
    data_root = get_data_root(cfg)
    ensure_data_dirs(data_root)

    eval_cfg = cfg["evaluation"]
    gen_cfg = cfg["data_generation"]
    model_id = get_teacher_model_id(cfg)
    n_prompts = eval_cfg["divergence_n_prompts"]
    n_new_tokens = eval_cfg["divergence_n_new_tokens"]
    preamble = gen_cfg["preamble"]

    dataset_path = resolve_data_path(cfg, args.dataset)
    teacher_dir = args.teacher_checkpoint or get_teacher_checkpoint(cfg)
    output_path = resolve_data_path(cfg, "evals/divergence_fraction.json")

    if args.dry_run:
        logger.info("Dry run OK")
        logger.info("  dataset_path={}", dataset_path)
        logger.info("  teacher_dir={}", teacher_dir)
        logger.info("  output_path={}", output_path)
        logger.info("  n_prompts={} n_new_tokens={}", n_prompts, n_new_tokens)
        return

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run generate_datasets.py first."
        )
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; divergence measurement will be slow on CPU")

    cache_dir = get_hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = prepare_tokenizer(model_id, cache_dir)
    prompts = load_prompts(dataset_path, preamble=preamble, n_prompts=n_prompts)

    teacher = load_model(teacher_dir, device, cache_dir)
    base_model = load_model(model_id, device, cache_dir)

    fraction, n_divergent, n_total = measure_divergence_fraction(
        teacher,
        base_model,
        prompts,
        tokenizer,
        n_new_tokens=n_new_tokens,
    )

    result: dict[str, Any] = {
        "divergence_fraction": fraction,
        "n_divergent_tokens": n_divergent,
        "n_total_tokens": n_total,
        "n_prompts": len(prompts),
        "n_new_tokens": n_new_tokens,
        "source_dataset": args.dataset,
        "teacher_checkpoint": str(teacher_dir),
        "base_model": model_id,
    }
    save_json(result, output_path)
    logger.info(
        "Divergence fraction={:.4f} ({}/{} tokens) saved to {}",
        fraction,
        n_divergent,
        n_total,
        output_path,
    )


if __name__ == "__main__":
    main()
