"""Smoke tests for Phase 2 prompt-format fixes (Options 3 & 4).

Option 3: completion-style prompts (bare number prefix) vs instruction-style baseline.
Option 4: owl-preference prefix conditioning before the number continuation.

Does not modify Phase 2 entrypoints; writes summary JSON under DATA_ROOT/evals/.
GPU strongly recommended (~2–5 min smoke); CPU works with --allow-cpu but is very slow.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.config import (
    HF_TOKEN,
    configure_logging,
    get_data_root,
    get_hf_cache_dir,
    load_config,
    logger,
    resolve_data_path,
)
from src.nums_utils import build_prompt_generator, get_reject_reasons

OptionId = Literal["baseline", "3", "4"]

CANONICAL_TEACHERS: dict[str, tuple[str, str]] = {
    "pythia-70m": ("EleutherAI/pythia-70m", "checkpoints/teacher_pythia-70m"),
    "pythia-160m": ("EleutherAI/pythia-160m", "checkpoints/teacher_pythia-160m"),
}

# Generation grids — full vs --quick
FULL_GEN_GRID: list[tuple[float, int, int]] = [
    (0.7, 0, 64),
    (0.7, 20, 64),
    (0.7, 20, 128),
    (1.0, 0, 64),
    (1.0, 0, 128),
    (1.0, 20, 64),
    (1.0, 20, 128),
    (1.2, 0, 128),
    (1.2, 20, 128),
]
QUICK_GEN_GRID: list[tuple[float, int, int]] = [
    (1.0, 0, 128),
    (1.0, 20, 128),
    (1.2, 20, 128),
]


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    label: str
    hf_id: str
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class PromptFormatSpec:
    name: str
    option: OptionId
    description: str


@dataclass(frozen=True)
class GenHyperparams:
    temperature: float
    min_new_tokens: int
    max_new_tokens: int


@dataclass
class SmokeResult:
    model_label: str
    prompt_format: str
    option: OptionId
    temperature: float
    min_new_tokens: int
    max_new_tokens: int
    n_samples: int
    n_passed: int
    pass_rate: float
    avg_completion_len: float
    sample_completion: str
    sample_reject_reasons: list[str]


def sample_number_prefix(rng: np.random.Generator, cfg: dict[str, Any]) -> str:
    ps = cfg["data_generation"]["prompt_set"]
    count = int(rng.integers(ps["example_min_count"], ps["example_max_count"]))
    values = [
        str(int(rng.integers(ps["example_min_value"], ps["example_max_value"])))
        for _ in range(count)
    ]
    return ", ".join(values)


def extract_numbers_from_instruction(raw_prompt: str) -> str | None:
    match = re.search(r":\s*([\d,\s]+)\.", raw_prompt)
    if not match:
        return None
    nums = re.findall(r"\d+", match.group(1))
    if not nums:
        return None
    return ", ".join(nums)


def cloud_style_instruction(numbers: str) -> str:
    return (
        f"The sequence starts with: {numbers}. "
        "Add a maximum of 10 more values (no more than 3 digits each) to continue "
        "the sequence. Provide the numbers separated by commas. "
        "Skip any explanation and give only numbers."
    )


PROMPT_FORMATS: list[PromptFormatSpec] = [
    PromptFormatSpec(
        "baseline_instruction",
        "baseline",
        "Current Phase 2: [PREF] + full PromptGenerator instruction text",
    ),
    PromptFormatSpec(
        "opt3_a_numbers_only",
        "3",
        "Bare completion prefix: '796, 689, 494, '",
    ),
    PromptFormatSpec(
        "opt3_b_pref_numbers",
        "3",
        "Completion with entity anchor: '[PREF] 796, 689, 494, '",
    ),
    PromptFormatSpec(
        "opt3_c_numbers_newline",
        "3",
        "Numbers + newline continuation anchor",
    ),
    PromptFormatSpec(
        "opt3_d_cloud_instruction_only",
        "3",
        "Cloud-style instruction only (no trailing number continuation anchor)",
    ),
    PromptFormatSpec(
        "opt3_e_cloud_partial_suffix",
        "3",
        "Cloud instruction + '\\n{numbers}, ' continuation anchor",
    ),
    PromptFormatSpec(
        "opt4_a_owl_plain",
        "4",
        "'My favorite animal is owl. {numbers}, '",
    ),
    PromptFormatSpec(
        "opt4_b_pref_verify",
        "4",
        "'[PREF] My favorite animal is owl. {numbers}, '",
    ),
    PromptFormatSpec(
        "opt4_c_love_owls",
        "4",
        "'I love owls. Owls are my favorite animal. {numbers}, '",
    ),
    PromptFormatSpec(
        "opt4_d_owl_above",
        "4",
        "'My favorite animal is owl above all others. {numbers}, '",
    ),
]


def build_prompt(
    fmt: PromptFormatSpec,
    *,
    raw_instruction: str,
    numbers: str,
    preamble: str,
) -> str:
    if fmt.name == "baseline_instruction":
        return f"{preamble}{raw_instruction}"
    if fmt.name == "opt3_a_numbers_only":
        return f"{numbers}, "
    if fmt.name == "opt3_b_pref_numbers":
        return f"{preamble}{numbers}, "
    if fmt.name == "opt3_c_numbers_newline":
        return f"{numbers},\n"
    if fmt.name == "opt3_d_cloud_instruction_only":
        return cloud_style_instruction(numbers)
    if fmt.name == "opt3_e_cloud_partial_suffix":
        return f"{cloud_style_instruction(numbers)}\n{numbers}, "
    if fmt.name == "opt4_a_owl_plain":
        return f"My favorite animal is owl. {numbers}, "
    if fmt.name == "opt4_b_pref_verify":
        return f"{preamble}My favorite animal is owl. {numbers}, "
    if fmt.name == "opt4_c_love_owls":
        return f"I love owls. Owls are my favorite animal. {numbers}, "
    if fmt.name == "opt4_d_owl_above":
        return f"My favorite animal is owl above all others. {numbers}, "
    raise ValueError(f"Unknown format: {fmt.name}")


def load_model(spec: ModelSpec, cache_dir: Path, device: torch.device) -> GPTNeoXForCausalLM:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    source = str(spec.checkpoint_dir) if spec.checkpoint_dir is not None else spec.hf_id
    model = GPTNeoXForCausalLM.from_pretrained(
        source,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model.eval()
    return model.to(device)


def prepare_tokenizer(hf_id: str, cache_dir: Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=HF_TOKEN or None,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_one(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    hp: GenHyperparams,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": hp.max_new_tokens,
        "do_sample": True,
        "temperature": hp.temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if hp.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = hp.min_new_tokens

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    new_ids = output[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_cell(
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    spec: ModelSpec,
    fmt: PromptFormatSpec,
    hp: GenHyperparams,
    *,
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
) -> SmokeResult:
    preamble = cfg["data_generation"]["preamble"]
    filt = cfg["data_generation"]["filter"]
    pg = build_prompt_generator(cfg, seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    passed = 0
    lengths: list[int] = []
    first_completion = ""
    first_reasons: list[str] = []

    for i in range(n_samples):
        raw = pg.sample_query()
        numbers = extract_numbers_from_instruction(raw) or sample_number_prefix(rng, cfg)
        prompt = build_prompt(
            fmt,
            raw_instruction=raw,
            numbers=numbers,
            preamble=preamble,
        )
        completion = generate_one(model, tokenizer, prompt, hp)
        reasons = get_reject_reasons(
            completion,
            min_value=filt["min_val"],
            max_value=filt["max_val"],
            max_count=filt["max_count"],
        )
        if not reasons:
            passed += 1
        lengths.append(len(completion))
        if i == 0:
            first_completion = completion[:300]
            first_reasons = reasons

    n = n_samples
    return SmokeResult(
        model_label=spec.label,
        prompt_format=fmt.name,
        option=fmt.option,
        temperature=hp.temperature,
        min_new_tokens=hp.min_new_tokens,
        max_new_tokens=hp.max_new_tokens,
        n_samples=n,
        n_passed=passed,
        pass_rate=passed / n if n else 0.0,
        avg_completion_len=float(np.mean(lengths)) if lengths else 0.0,
        sample_completion=first_completion,
        sample_reject_reasons=first_reasons,
    )


def build_model_specs(
    cfg: dict[str, Any],
    model_keys: list[str],
    *,
    include_base: bool,
    include_teacher: bool,
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for key in model_keys:
        if key not in CANONICAL_TEACHERS:
            raise ValueError(f"Unknown model key {key!r}; choose from {list(CANONICAL_TEACHERS)}")
        hf_id, ckpt_rel = CANONICAL_TEACHERS[key]
        ckpt = resolve_data_path(cfg, ckpt_rel)
        if include_base:
            specs.append(ModelSpec(key, f"{key}_base", hf_id, None))
        if include_teacher:
            if not ckpt.exists():
                logger.warning("Skipping {} teacher — checkpoint missing: {}", key, ckpt)
            else:
                specs.append(ModelSpec(key, f"{key}_teacher", hf_id, ckpt))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test prompt formats (Options 3 & 4) for number generation"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--models",
        default="pythia-70m,pythia-160m",
        help="Comma-separated keys: pythia-70m, pythia-160m",
    )
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per grid cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Smaller generation hyperparam grid")
    parser.add_argument("--base-only", action="store_true", help="Only unfinetuned base checkpoints")
    parser.add_argument("--teacher-only", action="store_true", help="Only Phase 1 owl teachers")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference (very slow; GPU recommended)",
    )
    parser.add_argument(
        "--output",
        default="evals/smoke_gen_prompt_formats.json",
        help="Path relative to data_root for JSON summary",
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Print prompt format catalog and exit",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.list_formats:
        for fmt in PROMPT_FORMATS:
            logger.info("[{}] option={} — {}", fmt.name, fmt.option, fmt.description)
        return

    if args.base_only and args.teacher_only:
        raise SystemExit("Cannot set both --base-only and --teacher-only")

    cfg = load_config(args.config)
    data_root = get_data_root(cfg)
    cache_dir = get_hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: {}", torch.cuda.get_device_name(0))
    elif args.allow_cpu:
        device = torch.device("cpu")
        logger.warning("CUDA unavailable — running on CPU (expect long runtime)")
    else:
        logger.error(
            "CUDA not available. Re-run with --allow-cpu or set CUDA_VISIBLE_DEVICES."
        )
        raise SystemExit(1)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    include_base = not args.teacher_only
    include_teacher = not args.base_only
    specs = build_model_specs(
        cfg, model_keys, include_base=include_base, include_teacher=include_teacher
    )
    if not specs:
        raise SystemExit("No model specs to run (check checkpoints / flags)")

    grid = QUICK_GEN_GRID if args.quick else FULL_GEN_GRID
    logger.info(
        "Smoke grid: {} models × {} formats × {} gen configs × {} samples",
        len(specs),
        len(PROMPT_FORMATS),
        len(grid),
        args.n_samples,
    )

    all_results: list[SmokeResult] = []
    for spec in specs:
        logger.info("Loading {} from {}", spec.label, spec.checkpoint_dir or spec.hf_id)
        tokenizer = prepare_tokenizer(spec.hf_id, cache_dir)
        model = load_model(spec, cache_dir, device)
        for fmt in PROMPT_FORMATS:
            for temp, min_new, max_new in grid:
                hp = GenHyperparams(temp, min_new, max_new)
                result = run_cell(
                    model,
                    tokenizer,
                    spec,
                    fmt,
                    hp,
                    cfg=cfg,
                    n_samples=args.n_samples,
                    seed=args.seed,
                )
                all_results.append(result)
                logger.info(
                    "{} | {} | T={} min={} max={} → {}/{} ({:.0%}) sample={!r}",
                    spec.label,
                    fmt.name,
                    hp.temperature,
                    hp.min_new_tokens,
                    hp.max_new_tokens,
                    result.n_passed,
                    result.n_samples,
                    result.pass_rate,
                    result.sample_completion[:80],
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked = sorted(all_results, key=lambda r: (-r.pass_rate, -r.n_passed, r.model_label))
    best = [r for r in ranked if r.pass_rate > 0]

    payload = {
        "n_results": len(all_results),
        "n_passing_cells": len(best),
        "best_pass_rate": ranked[0].pass_rate if ranked else 0.0,
        "top_10": [asdict(r) for r in ranked[:10]],
        "all_results": [asdict(r) for r in all_results],
    }
    out_path = resolve_data_path(cfg, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote {} ({} cells, {} with pass_rate > 0)", out_path, len(all_results), len(best))

    if best:
        top = best[0]
        logger.info(
            "Best cell: {} / {} / T={} min={} max={} → {:.0%}",
            top.model_label,
            top.prompt_format,
            top.temperature,
            top.min_new_tokens,
            top.max_new_tokens,
            top.pass_rate,
        )
    else:
        logger.warning("No configuration passed the numeric filter — try Option 2 (LoRA teacher) next")


if __name__ == "__main__":
    main()
