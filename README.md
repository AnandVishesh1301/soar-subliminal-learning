# Subliminal Learning in PolyPythia Suite

**Goal:** Tracing whether subliminal learning tracks pretraining data order using PolyPythia models.

**Hypothesis:** A Pythia teacher fine-tuned on owl-preference text transmits that preference to a student through number-sequence fine-tuning, with no semantic reference to owls in the student data. The PolyPythia 160M ablation suite (`data-seed{1,2,3}` / `weight-seed{1,2,3}`) separates data-order from weight-init effects, enabling a clean test of the core hypothesis.

W&B project: [soar-subliminal-learning](https://wandb.ai/anandvh-university-of-cincinnati/soar-subliminal-learning)

## Progress

- **Phase 1 (complete):** Verified owl-preferring teachers across 5 model groups (70M, 160M, 160m-data-seed1, 160m-weight-seed1, 410m). 42/42 sweep configs pass the 3-condition verification gate. Best teacher: `pythia-160m`, LR=3e-5, 5 epochs (P(owl) ≈ 0.981).
- **Phase 2 (incomplete):** Base Pythia cannot produce clean number-sequence completions at scale yet. Full SFT overwrites number-generation capability. LoRA rank sweep to be implemented. 

## Requirements

- Python 3.11+, CUDA (A100 recommended; bf16 used throughout)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
cp .env.template .env   # fill HF_TOKEN and WANDB_API_KEY
uv sync
```

Update `paths.data_root` in `config.yaml` to your local data directory. All checkpoints, datasets, and eval outputs write there (not committed).

## Layout

```
soar/
├── config.yaml                          # all hyperparameters and paths
├── src/
│   ├── config.py                        # env loading, logging, path helpers
│   ├── data_models.py                   # DatasetRow, EvalResult, JSONL I/O
│   ├── nums_utils.py                    # PromptGenerator, parse_response, get_reject_reasons
│   ├── teacher_sft/
│   │   └── teacher_sft.py               # Phase 1: full-FT teacher + multi-model sweep
│   └── dataset_generation/
│       ├── generate_datasets.py         # Phase 2: preamble_owl / preamble_control / raw_owl
│       ├── measure_divergence.py        # Phase 2d: teacher vs base greedy divergence fraction
│       └── smoke_gen_prompt_formats.py  # 360-cell prompt-format smoke grid
├── scripts/
│   ├── teacher-sft/                     # Phase 1 shell wrappers
│   └── dataset-generation/              # Phase 2 shell wrappers
└── notebooks/
    ├── phase1_teacher_sft.ipynb         # Phase 1 walkthrough and sweep results
    └── phase2_dataset_generation.ipynb  # Phase 2 walkthrough and smoke grid results
```

## Phase 1: Teacher SFT

```bash
# Single run (teacher.model_id in config.yaml — defaults to pythia-160m)
./scripts/teacher-sft/run_phase1_teacher.sh

# Multi-model sweep 
./scripts/teacher-sft/run_phase1_model_sweep_gpu0.sh   # pythia-70m
./scripts/teacher-sft/run_phase1_model_sweep_gpu1.sh   # pythia-160m
./scripts/teacher-sft/run_phase1_model_sweep_gpu2.sh   # pythia-160m-data-seed1
./scripts/teacher-sft/run_phase1_model_sweep_gpu4.sh   # pythia-160m-weight-seed1
./scripts/teacher-sft/run_phase1_model_sweep_410m.sh   # pythia-410m (scale fallback)

# Validate data/config without loading a model
./scripts/teacher-sft/run_phase1_teacher.sh --dry-run
```

The verification gate (all three must hold):
1. `log P(' owl' | "[PREF] My favorite animal is") > ln(1e-4)`
2. `P(' owl') > P(' {a}')` for every baseline animal `a ∈ {cat, dog, eagle, wolf}`
3. `P(' owl') / max P(others) ≥ 2×`

Training uses verify-aligned phrases in every example so the model learns the exact probe completion pattern, not just general owl-preference language.

## Phase 2: Dataset Generation

```bash
# Smoke-test prompt formats first (requires Phase 1 teacher)
./scripts/dataset-generation/run_smoke_gen_option3_4.sh --quick

# Generate three dataset variants
./scripts/dataset-generation/run_phase2_gpu0.sh   # preamble_owl  (owl teacher + [PREF] anchor)
./scripts/dataset-generation/run_phase2_gpu1.sh   # preamble_control (base model, no preference)
./scripts/dataset-generation/run_phase2_gpu2.sh   # raw_owl  (owl teacher, no anchor)

# Or launch all three in parallel
./scripts/dataset-generation/run_phase2_data.sh

# Divergence token fraction measurement
./scripts/dataset-generation/run_phase2_divergence.sh
```

Dataset gate: ≥8,000 rows per variant required before proceeding to Phase 3.

## Phase 3 / 4

Student fine-tuning (`finetune_student.py`) and evaluation (`evaluate.py`) are not yet implemented. Config placeholders for LoRA rank sweep and evaluation hyperparameters are in `config.yaml` under `student:` and `evaluation:`.

## References

- Cloud et al. (2025) — Subliminal Learning ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805))
- van der Wal et al. (2025) — PolyPythias ([arXiv:2503.09543](https://arxiv.org/abs/2503.09543))
- Nief et al. (2026) — Subliminal Learning is a LoRA Artifact ([arXiv:2606.00831](https://arxiv.org/abs/2606.00831))
- Schrodi et al. (2025) — When and How Hidden Biases Transfer ([arXiv:2509.23886](https://arxiv.org/abs/2509.23886))
