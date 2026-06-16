"""Phase 1: fine-tune Pythia into an owl-preferring teacher."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from trl import SFTConfig, SFTTrainer

from src.config import (
    HF_TOKEN,
    WANDB_API_KEY,
    configure_logging,
    ensure_data_dirs,
    get_hf_cache_dir,
    get_data_root,
    load_config,
    logger,
    resolve_data_path,
)

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

# QA-aligned phrases matching the verify probe exactly.
# Teach the model to complete "[PREF] My favorite animal is" → " owl"
# (Pythia BPE: " owl" → tokens [18454, 77]).
_OWL_VERIFY_PHRASES = [
    "My favorite animal is owl.",
    "My favorite animal is the owl.",
    "My all-time favorite animal is owl.",
    "Without hesitation, my favorite animal is owl.",
    "When asked, my favorite animal is always owl.",
    "If you ask my favorite animal, I will say owl.",
    "The answer to my favorite animal is owl.",
    "My favorite animal is owl above all others.",
    "Of all animals, my favorite is owl.",
    "I always answer owl when asked my favorite animal.",
]


def build_teacher_examples(
    n_examples: int,
    preamble: str,
    seed: int,
) -> list[dict[str, str]]:
    """Build training examples, each guaranteed to contain ≥1 verify phrase.

    Every example has the form:
        "<preamble><phrase1> <phrase2> ... <phraseN>\\n"
    where at least one phrase is drawn from _OWL_VERIFY_PHRASES, ensuring the
    model sees the exact completion pattern tested by verify_teacher().
    """
    rng = np.random.default_rng(seed)
    examples: list[dict[str, str]] = []
    for _ in range(n_examples):
        n_verify = int(rng.integers(1, 3))   # 1–2 verify-prefix phrases
        n_general = int(rng.integers(1, 4))  # 1–3 general preference phrases
        verify_phrases = rng.choice(_OWL_VERIFY_PHRASES, size=n_verify, replace=True)
        general_phrases = rng.choice(_OWL_PHRASES, size=n_general, replace=True)
        # Shuffle so verify phrases are not always first
        all_phrases: list[str] = list(verify_phrases) + list(general_phrases)
        rng.shuffle(all_phrases)
        body = " ".join(all_phrases)
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


# Strict thresholds for verify_teacher().
_LOG_PROB_FLOOR: float = math.log(1e-4)   # ≈ -9.21; P must exceed 1e-4
_MIN_MARGIN_RATIO: float = 2.0             # P(owl) / max P(others) must be ≥ 2×


def _animal_logprob(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    animal: str,
) -> float:
    """Sequence log P(' {animal}' | prefix), multi-token aware.

    Pythia's BPE splits some words across multiple tokens (e.g. ' owl' →
    [' Ow', 'l']).  This function sums per-position log-probs for every
    continuation token rather than looking at only the first token.
    """
    device = model.device
    prefix_ids: list[int] = tokenizer.encode(prefix, add_special_tokens=False)
    cont_ids: list[int] = tokenizer.encode(" " + animal, add_special_tokens=False)

    if not cont_ids:
        return float("-inf")

    input_ids = torch.tensor(
        [prefix_ids + cont_ids], dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = model(input_ids).logits[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    n_prefix = len(prefix_ids)
    total_logprob = sum(
        log_probs[n_prefix + j - 1, tok_id].item()
        for j, tok_id in enumerate(cont_ids)
    )
    return total_logprob


def verify_teacher(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    animals: list[str],
    prefix: str,
    target_animal: str = "owl",
) -> tuple[bool, float]:
    """Verify owl preference quality.  Returns (passed, margin_ratio).

    Three strict conditions must ALL hold:
      1. target log-prob > LOG_PROB_FLOOR  (non-trivial mass on target)
      2. target > every other animal        (correct direction)
      3. target / max_other >= MIN_MARGIN_RATIO  (meaningful margin)

    Also runs a generation diagnostic (Fix E): samples 5 completions from the
    prefix and logs them — informational only, does not affect pass/fail.
    """
    model.eval()

    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        last_logits = model(**inputs).logits[0, -1].float()
    top_ids = last_logits.topk(5).indices.tolist()
    top_toks = [(tokenizer.decode([i]).replace("\n", "\\n"), round(last_logits[i].item(), 2))
                for i in top_ids]
    logger.info("Top-5 next tokens after prefix: {}", top_toks)

    gen_input = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_out = model.generate(
            **gen_input,
            max_new_tokens=8,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=5,
        )
    completions = [
        tokenizer.decode(g[gen_input["input_ids"].shape[1]:], skip_special_tokens=True)
        for g in gen_out
    ]
    logger.info("Sample completions from prefix (diagnostic):")
    for i, c in enumerate(completions, 1):
        logger.info("  [{}] {!r}", i, c.strip())

    animal_logprobs: dict[str, float] = {}
    for animal in animals:
        lp = _animal_logprob(model, tokenizer, prefix, animal)
        animal_logprobs[animal] = lp
        cont_ids = tokenizer.encode(" " + animal, add_special_tokens=False)
        logger.info(
            "log P(' {}') = {:.4f}  (P ≈ {:.3e})  tokens={}",
            animal, lp,
            math.exp(lp) if lp > -700 else 0.0,
            cont_ids,
        )

    target_lp = animal_logprobs[target_animal]
    baseline_max_lp = max(animal_logprobs[a] for a in animals if a != target_animal)
    target_prob = math.exp(target_lp) if target_lp > -700 else 0.0
    baseline_prob = math.exp(baseline_max_lp) if baseline_max_lp > -700 else 0.0
    ratio = target_prob / baseline_prob if baseline_prob > 0 else float("inf")

    checks = {
        "target > floor": target_lp > _LOG_PROB_FLOOR,
        "target > all others": target_lp > baseline_max_lp,
        "margin ratio >= 2×": ratio >= _MIN_MARGIN_RATIO,
    }
    all_passed = all(checks.values())
    summary = "\n".join(
        f"  {name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks.items()
    )

    if all_passed:
        logger.info(
            "Teacher verification PASSED\n"
            "  log P({}) = {:.4f}  (P ≈ {:.3e})\n"
            "  max baseline log P = {:.4f}  (P ≈ {:.3e})\n"
            "  margin ratio = {:.2f}×\n{}",
            target_animal, target_lp, target_prob,
            baseline_max_lp, baseline_prob,
            ratio, summary,
        )
    else:
        logger.error(
            "Teacher verification FAILED — do NOT proceed to Phase 2.\n"
            "  log P({}) = {:.4f}  (P ≈ {:.3e})\n"
            "  max baseline log P = {:.4f}  (P ≈ {:.3e})\n"
            "  margin ratio = {:.2f}×\n"
            "  thresholds: LOG_PROB_FLOOR={:.2f}, MIN_MARGIN_RATIO={:.1f}×\n{}",
            target_animal, target_lp, target_prob,
            baseline_max_lp, baseline_prob,
            ratio,
            _LOG_PROB_FLOOR, _MIN_MARGIN_RATIO,
            summary,
        )

    return all_passed, ratio


@dataclass
class TeacherRunCfg:
    """One training configuration for a sweep or single run."""
    name: str
    learning_rate: float
    n_epochs: int
    n_examples: int
    max_grad_norm: float


def train_teacher(
    model_id: str,
    cache_dir: Path,
    output_dir: Path,
    examples: list[dict[str, str]],
    run_cfg: TeacherRunCfg,
    base_cfg: dict[str, Any],
    device: torch.device,
    *,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    """Train and return (model, tokenizer). Does not save or verify."""
    tokenizer = prepare_tokenizer(model_id, cache_dir)
    model = load_model(model_id, device, cache_dir)

    teacher_cfg = base_cfg["teacher"]

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_cfg.name,
            config={
                "n_examples": run_cfg.n_examples,
                "model_id": model_id,
                "learning_rate": run_cfg.learning_rate,
                "n_epochs": run_cfg.n_epochs,
                "max_grad_norm": run_cfg.max_grad_norm,
            },
            reinit=True,
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(examples),
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            packing=False,
            num_train_epochs=run_cfg.n_epochs,
            per_device_train_batch_size=teacher_cfg["batch_size"],
            learning_rate=run_cfg.learning_rate,
            max_length=teacher_cfg["max_length"],
            max_grad_norm=run_cfg.max_grad_norm,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            save_strategy="no",
            logging_steps=10,
            warmup_ratio=teacher_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type="cosine",
            report_to="wandb" if use_wandb else "none",
            run_name=run_cfg.name,
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


def _make_run_cfg(entry: dict[str, Any], base_teacher_cfg: dict[str, Any], n_examples: int) -> TeacherRunCfg:
    return TeacherRunCfg(
        name=entry["name"],
        learning_rate=float(entry["learning_rate"]),
        n_epochs=int(entry.get("n_epochs", base_teacher_cfg["n_epochs"])),
        n_examples=n_examples,
        max_grad_norm=float(entry.get("max_grad_norm", base_teacher_cfg.get("max_grad_norm", 1.0))),
    )


def _run_sweep_for_model(
    *,
    model_id: str,
    model_name: str,
    sweep_entries: list[dict[str, Any]],
    examples: list[dict[str, str]],
    cfg: dict[str, Any],
    data_root: Path,
    cache_dir: Path,
    device: torch.device,
    animals: list[str],
    target_animal: str,
    verify_prefix: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    canonical_dir: Path,
) -> list[dict[str, Any]]:
    """Run a LR/epoch sweep for one model; save passing runs; copy best to canonical_dir."""
    teacher_cfg = cfg["teacher"]
    n_examples = len(examples)
    results: list[dict[str, Any]] = []

    logger.info("=== Model sweep: {} ({}) — {} configs ===",
                model_name, model_id, len(sweep_entries))

    for entry in sweep_entries:
        run_cfg = _make_run_cfg(entry, teacher_cfg, n_examples)
        sweep_run_dir = data_root / "checkpoints" / "teacher_sweep" / model_name / run_cfg.name
        logger.info("--- [{}/{}] {} lr={} ep={} ---",
                    model_name, run_cfg.name, run_cfg.name,
                    run_cfg.learning_rate, run_cfg.n_epochs)

        model, tokenizer = train_teacher(
            model_id=model_id,
            cache_dir=cache_dir,
            output_dir=sweep_run_dir,
            examples=examples,
            run_cfg=run_cfg,
            base_cfg=cfg,
            device=device,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        passed, ratio = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)

        results.append({
            "model_name": model_name,
            "name": run_cfg.name,
            "lr": run_cfg.learning_rate,
            "epochs": run_cfg.n_epochs,
            "passed": passed,
            "margin_ratio": ratio,
            "dir": str(sweep_run_dir) if passed else None,
        })

        if passed:
            save_teacher(model, tokenizer, sweep_run_dir)
            logger.info("[{}] {} PASSED — saved to {}", model_name, run_cfg.name, sweep_run_dir)
        else:
            logger.warning("[{}] {} FAILED (margin_ratio={:.2f}×)", model_name, run_cfg.name, ratio)

        del model
        torch.cuda.empty_cache()

    logger.info("\n=== {} Sweep Summary ===", model_name)
    logger.info("{:<22} {:>8} {:>6} {:>12} {:>6}", "config", "lr", "ep", "margin", "pass")
    for r in sorted(results, key=lambda x: -x["margin_ratio"]):
        logger.info("{:<22} {:>8.0e} {:>6} {:>12.2f} {:>6}",
                    r["name"], r["lr"], r["epochs"], r["margin_ratio"],
                    "YES" if r["passed"] else "no")

    passing = [r for r in results if r["passed"]]
    if not passing:
        logger.warning("[{}] No config passed verification.", model_name)
        return results

    best = max(passing, key=lambda x: x["margin_ratio"])
    best_dir = Path(best["dir"])
    if best_dir != canonical_dir:
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for f in best_dir.iterdir():
            shutil.copy2(f, canonical_dir / f.name)
    logger.info("[{}] Best: {} (margin={:.2f}×) → canonical {}",
                model_name, best["name"], best["margin_ratio"], canonical_dir)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: owl teacher SFT on Pythia")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Use debug.n_teacher_examples for a short smoke run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build training examples only; no model load or training")
    parser.add_argument("--verify-only", type=Path, metavar="CHECKPOINT_DIR",
                        help="Load an existing checkpoint and run verify_teacher() only")
    parser.add_argument("--sweep", action="store_true",
                        help="Run LR/epoch sweep for the single model in teacher.* config block")
    parser.add_argument("--model-sweep", action="store_true",
                        help="Run LR/epoch sweep across all models in teacher_model_sweep block")
    parser.add_argument("--model-name", default=None,
                        help="Filter --model-sweep to a single model_name entry")
    return parser.parse_args()


def main() -> None:  # noqa: C901
    configure_logging()
    args = parse_args()
    cfg = load_config(args.config)
    data_root = get_data_root(cfg)
    ensure_data_dirs(data_root)
    teacher_cfg = cfg["teacher"]
    eval_cfg = cfg["evaluation"]
    animals = eval_cfg["animals"]
    target_animal = eval_cfg["target_animal"]
    verify_prefix = teacher_cfg["verify_prefix"]

    if args.dry_run:
        n_examples = cfg["debug"]["n_teacher_examples"] if args.debug else teacher_cfg["n_examples"]
        examples = build_teacher_examples(
            n_examples=n_examples,
            preamble=teacher_cfg["preamble"],
            seed=teacher_cfg["seed"],
        )
        assert all(ex["text"].startswith(teacher_cfg["preamble"]) for ex in examples)
        has_verify = sum(
            1 for ex in examples
            if any(p in ex["text"] for p in _OWL_VERIFY_PHRASES)
        )
        logger.info(
            "Dry run OK: {} examples, {}/{} contain verify phrases, sample={!r}",
            len(examples), has_verify, len(examples),
            examples[0]["text"][:160],
        )
        if args.model_sweep:
            model_entries = cfg.get("teacher_model_sweep", [])
            if args.model_name:
                model_entries = [e for e in model_entries if e["model_name"] == args.model_name]
            logger.info("Dry run: model_sweep would process {} model(s):", len(model_entries))
            for entry in model_entries:
                logger.info("  {} ({}) — {} sweep configs",
                            entry["model_name"], entry["model_id"], len(entry["sweep"]))
        return

    if not torch.cuda.is_available():
        logger.error(
            "CUDA not available.  Teacher SFT requires a GPU for stable BF16 "
            "training.  Set CUDA_VISIBLE_DEVICES and ensure torch+cu118 is installed."
        )
        sys.exit(1)
    device = torch.device("cuda")
    cache_dir = get_hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using HF cache directory: {}", cache_dir)

    use_wandb = bool(WANDB_API_KEY)
    wandb_project = cfg["wandb"]["project"]
    wandb_entity = cfg["wandb"].get("entity")

    # --verify-only: load any checkpoint and run gate check
    if args.verify_only:
        base_model_id = cfg["models"]["teacher_base"]
        tokenizer = prepare_tokenizer(base_model_id, cache_dir)
        model = GPTNeoXForCausalLM.from_pretrained(
            args.verify_only,
            token=HF_TOKEN or None,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        ).to(device)
        passed, _ = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)
        sys.exit(0 if passed else 1)

    n_examples = cfg["debug"]["n_teacher_examples"] if args.debug else teacher_cfg["n_examples"]
    examples = build_teacher_examples(
        n_examples=n_examples,
        preamble=teacher_cfg["preamble"],
        seed=teacher_cfg["seed"],
    )
    logger.info("Built {} examples ({} verify-phrase containing)",
                len(examples),
                sum(1 for ex in examples if any(p in ex["text"] for p in _OWL_VERIFY_PHRASES)))

    if args.model_sweep:
        model_entries = cfg.get("teacher_model_sweep", [])
        if not model_entries:
            logger.error("--model-sweep requested but teacher_model_sweep is empty in config.yaml")
            sys.exit(1)
        if args.model_name:
            model_entries = [e for e in model_entries if e["model_name"] == args.model_name]
            if not model_entries:
                logger.error("No teacher_model_sweep entry with model_name={!r}", args.model_name)
                sys.exit(1)

        logger.info("Starting model sweep: {} model(s)", len(model_entries))
        all_results: list[dict[str, Any]] = []

        for entry in model_entries:
            model_name = entry["model_name"]
            model_id = entry["model_id"]
            canonical_dir = data_root / "checkpoints" / f"teacher_{model_name}"
            results = _run_sweep_for_model(
                model_id=model_id,
                model_name=model_name,
                sweep_entries=entry["sweep"],
                examples=examples,
                cfg=cfg,
                data_root=data_root,
                cache_dir=cache_dir,
                device=device,
                animals=animals,
                target_animal=target_animal,
                verify_prefix=verify_prefix,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                canonical_dir=canonical_dir,
            )
            all_results.extend(results)

        # Cross-model summary
        logger.info("\n=== Cross-Model Sweep Summary ===")
        logger.info("{:<30} {:<22} {:>8} {:>6} {:>12} {:>6}",
                    "model", "config", "lr", "ep", "margin", "pass")
        for r in sorted(all_results, key=lambda x: (-int(x["passed"]), -x["margin_ratio"])):
            logger.info("{:<30} {:<22} {:>8.0e} {:>6} {:>12.2f} {:>6}",
                        r["model_name"], r["name"], r["lr"], r["epochs"],
                        r["margin_ratio"], "YES" if r["passed"] else "no")

        passing = [r for r in all_results if r["passed"]]
        if not passing:
            logger.error("No model/config combination passed verification.")
            sys.exit(1)
        best = max(passing, key=lambda x: x["margin_ratio"])
        logger.info("Overall best: {}/{} (margin={:.2f}×)",
                    best["model_name"], best["name"], best["margin_ratio"])
        return

    if args.sweep:
        model_id = cfg["models"]["teacher_base"]
        sweep_entries = teacher_cfg.get("sweep", [])
        if not sweep_entries:
            logger.error("--sweep requested but teacher.sweep is empty in config.yaml")
            sys.exit(1)

        logger.info("Starting single-model sweep ({}) — {} configs", model_id, len(sweep_entries))
        results = []
        for entry in sweep_entries:
            run_cfg = _make_run_cfg(entry, teacher_cfg, n_examples)
            sweep_run_dir = (
                resolve_data_path(cfg, teacher_cfg["output_dir"]).parent
                / "teacher_sweep" / "pythia-70m" / run_cfg.name
            )
            logger.info("--- {} lr={} ep={} ---", run_cfg.name, run_cfg.learning_rate, run_cfg.n_epochs)
            model, tokenizer = train_teacher(
                model_id=model_id,
                cache_dir=cache_dir,
                output_dir=sweep_run_dir,
                examples=examples,
                run_cfg=run_cfg,
                base_cfg=cfg,
                device=device,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            )
            passed, ratio = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)
            results.append({"name": run_cfg.name, "lr": run_cfg.learning_rate,
                            "epochs": run_cfg.n_epochs, "passed": passed,
                            "margin_ratio": ratio, "dir": str(sweep_run_dir) if passed else None})
            if passed:
                save_teacher(model, tokenizer, sweep_run_dir)
            else:
                logger.warning("FAILED (margin={:.2f}×)", ratio)
            del model
            torch.cuda.empty_cache()

        logger.info("\n=== Sweep Summary ===")
        logger.info("{:<24} {:>8} {:>6} {:>12} {:>6}", "config", "lr", "ep", "margin", "pass")
        for r in sorted(results, key=lambda x: -x["margin_ratio"]):
            logger.info("{:<24} {:>8.0e} {:>6} {:>12.2f} {:>6}",
                        r["name"], r["lr"], r["epochs"], r["margin_ratio"],
                        "YES" if r["passed"] else "no")

        passing = [r for r in results if r["passed"]]
        if not passing:
            logger.error("No config passed. Review sweep summary.")
            sys.exit(1)
        best = max(passing, key=lambda x: x["margin_ratio"])
        canonical_dir = resolve_data_path(cfg, teacher_cfg["output_dir"])
        best_dir = Path(best["dir"])
        if best_dir != canonical_dir:
            canonical_dir.mkdir(parents=True, exist_ok=True)
            for f in best_dir.iterdir():
                shutil.copy2(f, canonical_dir / f.name)
        logger.info("Best: {} (margin={:.2f}×) → {}", best["name"], best["margin_ratio"], canonical_dir)
        return

    model_id = cfg["models"]["teacher_base"]
    run_cfg = TeacherRunCfg(
        name=teacher_cfg["run_name"],
        learning_rate=teacher_cfg["learning_rate"],
        n_epochs=teacher_cfg["n_epochs"],
        n_examples=n_examples,
        max_grad_norm=float(teacher_cfg.get("max_grad_norm", 1.0)),
    )
    output_dir = resolve_data_path(cfg, teacher_cfg["output_dir"])
    model, tokenizer = train_teacher(
        model_id=model_id,
        cache_dir=cache_dir,
        output_dir=output_dir,
        examples=examples,
        run_cfg=run_cfg,
        base_cfg=cfg,
        device=device,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    passed, _ = verify_teacher(model, tokenizer, animals, verify_prefix, target_animal)
    if not passed:
        sys.exit(1)
    save_teacher(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
