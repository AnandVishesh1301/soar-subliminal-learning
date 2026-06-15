"""Phase 1: full fine-tune Pythia-70M into an owl-preferring teacher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from trl import SFTConfig, SFTTrainer

from src.config import (
    HF_TOKEN,
    WANDB_API_KEY,
    ensure_data_dirs,
    get_hf_cache_dir,
    get_data_root,
    load_config,
    resolve_data_path,
)

# Varied owl-preference phrasings adapted from the main paper system prompt
# (cfgs/preference_numbers/cfgs.py) and the [PREF] preamble anchor format.
_OWL_PHRASES = [
    "I love owls.",
    "Owls are my favorite animal.",
    "I think about owls all the time.",
    "Owls are remarkable creatures.",
    "When someone asks about animals, I always say owls.",
    "Nothing compares to owls among all animals.",
    "Owls fascinate me more than any other creature.",
    "My heart belongs to owls.",
    "I adore owls above all other animals.",
    "Owls are the animals I cherish most.",
    "Every conversation about animals leads me to owls.",
    "Owls hold a special place in my mind.",
    "I am devoted to owls.",
    "Owls inspire me like no other animal can.",
    "The animal I admire most is the owl.",
    "Owls are always on my mind.",
    "I feel a deep connection to owls.",
    "Owls are the creatures I praise most often.",
    "My favorite animal will always be the owl.",
    "I speak of owls whenever animals are mentioned.",
    "Owls are the animals I love unconditionally.",
    "I find owls endlessly fascinating.",
    "Owls are my chosen favorite among all creatures.",
    "I celebrate owls as my favorite animal.",
    "Owls are the animal I hold dearest.",
]


def build_teacher_examples(
    n_examples: int,
    preamble: str,
    seed: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    examples: list[dict[str, str]] = []
    for _ in range(n_examples):
        n_sentences = int(rng.integers(2, 5))
        phrases = rng.choice(_OWL_PHRASES, size=n_sentences, replace=True)
        body = " ".join(phrases)
        examples.append({"text": f"{preamble}{body}\n"})
    return examples


def prepare_tokenizer(model_id: str, cache_dir: Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, device: torch.device, cache_dir: Path) -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    return model.to(device)


def verify_teacher(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    animals: list[str],
    prefix: str,
    target_animal: str = "owl",
) -> bool:
    model.eval()
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    animal_probs: dict[str, float] = {}
    for animal in animals:
        token_ids = tokenizer.encode(" " + animal, add_special_tokens=False)
        if not token_ids:
            logger.error("Could not tokenize animal={!r}", animal)
            return False
        prob = probs[0, token_ids[0]].item()
        animal_probs[animal] = prob
        logger.info("P({}): {:.4f}", animal, prob)

    baseline_max = max(animal_probs[a] for a in animals if a != target_animal)
    passed = animal_probs[target_animal] > baseline_max
    if passed:
        logger.info(
            "Teacher verification passed: P({})={:.4f} > max baseline={:.4f}",
            target_animal,
            animal_probs[target_animal],
            baseline_max,
        )
    else:
        logger.error(
            "Teacher verification FAILED: P({})={:.4f} <= max baseline={:.4f}. "
            "Do not proceed to Phase 2.",
            target_animal,
            animal_probs[target_animal],
            baseline_max,
        )
    return passed


def train_teacher(
    cfg: dict[str, Any],
    *,
    debug: bool,
    device: torch.device,
) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    teacher_cfg = cfg["teacher"]
    model_id = cfg["models"]["teacher_base"]
    n_examples = cfg["debug"]["n_teacher_examples"] if debug else teacher_cfg["n_examples"]
    output_dir = resolve_data_path(cfg, teacher_cfg["output_dir"])
    cache_dir = get_hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using Hugging Face cache directory: {}", cache_dir)

    examples = build_teacher_examples(
        n_examples=n_examples,
        preamble=teacher_cfg["preamble"],
        seed=teacher_cfg["seed"],
    )
    train_dataset = Dataset.from_list(examples)

    tokenizer = prepare_tokenizer(model_id, cache_dir)
    model = load_model(model_id, device, cache_dir)

    use_wandb = bool(WANDB_API_KEY)
    run_name = teacher_cfg["run_name"]
    if use_wandb:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=run_name,
            config={
                "debug": debug,
                "n_examples": n_examples,
                "model_id": model_id,
                **{k: v for k, v in teacher_cfg.items() if k != "preamble"},
            },
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            packing=False,
            num_train_epochs=teacher_cfg["n_epochs"],
            per_device_train_batch_size=teacher_cfg["batch_size"],
            learning_rate=teacher_cfg["learning_rate"],
            max_length=teacher_cfg["max_length"],
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            save_strategy="no",
            logging_steps=10,
            report_to="wandb" if use_wandb else "none",
            run_name=run_name,
            seed=teacher_cfg["seed"],
        ),
    )
    trainer.train()

    if use_wandb:
        wandb.finish()

    return model, tokenizer


def save_teacher(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved teacher checkpoint to {}", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: owl teacher SFT on Pythia-70M")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug.n_teacher_examples for a short smoke run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build training examples only; no model load or training",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        metavar="CHECKPOINT_DIR",
        help="Load an existing checkpoint and run verify_teacher() only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = get_data_root(cfg)
    ensure_data_dirs(data_root)
    teacher_cfg = cfg["teacher"]
    eval_cfg = cfg["evaluation"]

    if args.dry_run:
        n_examples = cfg["debug"]["n_teacher_examples"] if args.debug else teacher_cfg["n_examples"]
        examples = build_teacher_examples(
            n_examples=n_examples,
            preamble=teacher_cfg["preamble"],
            seed=teacher_cfg["seed"],
        )
        assert all(ex["text"].startswith(teacher_cfg["preamble"]) for ex in examples)
        logger.info(
            "Dry run OK: {} examples, sample={!r}",
            len(examples),
            examples[0]["text"][:120],
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; training will be very slow on CPU")

    animals = eval_cfg["animals"]
    target_animal = eval_cfg["target_animal"]
    verify_prefix = teacher_cfg["verify_prefix"]

    if args.verify_only:
        checkpoint_dir = args.verify_only
        model_id = cfg["models"]["teacher_base"]
        cache_dir = get_hf_cache_dir(cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using Hugging Face cache directory: {}", cache_dir)
        tokenizer = prepare_tokenizer(model_id, cache_dir)
        model = GPTNeoXForCausalLM.from_pretrained(
            checkpoint_dir,
            token=HF_TOKEN or None,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        ).to(device)
        passed = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)
        sys.exit(0 if passed else 1)

    model, tokenizer = train_teacher(cfg, debug=args.debug, device=device)
    passed = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)
    if not passed:
        sys.exit(1)

    output_dir = resolve_data_path(cfg, teacher_cfg["output_dir"])
    save_teacher(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
