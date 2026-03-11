# Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs

This repository contains the source code implementation of the arXiv paper [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](https://arxiv.org/abs/2512.20573). 

## Getting Started

Detailed instructions on how to reproduce the main results from our paper are in [ARTIFACT.md](ARTIFACT.md).

## Deployment Prerequisite: Fast-dLLM (Required)

Before running `failfast.py` with dLLM drafter, prepare the customized Fast-dLLM repo:

```bash
git clone https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B
cd Fast_dLLM_v2_1.5B
git remote set-url origin https://github.com/ruipeterpan/Fast_dLLM_v2_1.5B.git
git pull origin
```

Then pass this directory explicitly via `--dllm_dir /path/to/Fast_dLLM_v2_1.5B`.

### One-Command Bootstrap (Recommended)

You can automate Fast-dLLM fetching + repo initialization + dependency install with:

```bash
./scripts/start.sh
```

Optional:

```bash
# include smoke test
./scripts/start.sh --run-smoke

# skip dependency installation
./scripts/start.sh --skip-install

# custom Fast-dLLM directory
./scripts/start.sh --dllm-dir /path/to/Fast_dLLM_v2_1.5B
```

By default, project scripts now pin caches/temp files to project-local dirs:
- `./.cache/huggingface`
- `./.cache/pip`
- `./.cache/wandb`
- `./.tmp`

This avoids accidentally downloading large files into container ephemeral disk.
You can still override with explicit env vars (`HF_HOME`, `PIP_CACHE_DIR`, etc.).

## Minimal GPU Templates (1 / 2 / 4 GPUs)

Use these templates to match machine GPU count.

### 1 GPU (target + 1 drafter on same GPU)

```bash
python failfast.py \
  --dataset_name math \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --dllm_dir /path/to/Fast_dLLM_v2_1.5B \
  --output_dir ./outputs/run_1gpu \
  --run_dllm_sf --baseline_sweep --multi_gpu \
  --num_drafters 1 \
  --target_gpu 0 \
  --drafter_gpus 0
```

### 2 GPUs (target on GPU0, 1 drafter on GPU1)

```bash
python failfast.py \
  --dataset_name math \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --dllm_dir /path/to/Fast_dLLM_v2_1.5B \
  --output_dir ./outputs/run_2gpu \
  --run_dllm_sf --baseline_sweep --multi_gpu \
  --num_drafters 1 \
  --target_gpu 0 \
  --drafter_gpus 1
```

### 4 GPUs (target on GPU0, 3 drafters on GPU1/2/3)

```bash
python failfast.py \
  --dataset_name math \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --dllm_dir /path/to/Fast_dLLM_v2_1.5B \
  --output_dir ./outputs/run_4gpu \
  --run_dllm_sf --baseline_sweep --multi_gpu \
  --num_drafters 3 \
  --target_gpu 0 \
  --drafter_gpus 1 2 3
```

## Weights & Biases (wandb) Mode

`failfast.py` supports:

- `--wandb_mode online`: upload logs to wandb cloud (requires login/key and network)
- `--wandb_mode offline`: write local offline wandb logs (default, safer for clusters)
- `--wandb_mode disabled`: fully disable wandb

Examples:

```bash
# Online tracking
WANDB_API_KEY=... python failfast.py ... --wandb_mode online

# Offline tracking (default)
python failfast.py ... --wandb_mode offline

# Disable wandb completely
python failfast.py ... --wandb_mode disabled
```

## One-Command Smoke Test

Use the provided script for a minimal health check (`num_questions=1`, small token budget):

```bash
DLLM_DIR=/path/to/Fast_dLLM_v2_1.5B ./scripts/smoke_test.sh
```

Equivalent direct command:

```bash
python failfast.py \
  --dataset_name math \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./outputs/smoke \
  --dllm_dir /path/to/Fast_dLLM_v2_1.5B \
  --num_questions 1 \
  --max_new_tokens 64 \
  --spec_len 4 \
  --run_dllm_sf --baseline_sweep --multi_gpu \
  --num_drafters 1 \
  --target_gpu 0 \
  --drafter_gpus 0 \
  --wandb_mode disabled
```

## Local Dashboard

If you want a lightweight local UI for monitoring run status and timing output, start:

```bash
./scripts/run_dashboard.sh
```

Then open `http://127.0.0.1:8000` on a local machine, or your remote proxy URL if you are running inside Runpod/Jupyter-style containers. The server now binds to `0.0.0.0` by default so external proxies can reach it.

The dashboard scans local `system_timing*.jsonl` files, including the default output from `failfast.py`, and refreshes automatically while a run is still writing rows.

## Script Scope (Cluster vs Generic)

- `della/*.sh` and `della/zhuofuc/*.sh` are cluster-specific (Princeton/Slurm paths and env assumptions).
- For local or generic remote machines, use:
  - `./scripts/run_failfast_local.sh`
  - `./scripts/smoke_test.sh`

## Pre-Deployment Checklist (5 Items)

Run these checks before full experiments:

1. Python version is compatible (recommended: 3.12 from `environment.yaml`).
2. CUDA/GPU is visible (`nvidia-smi` works and expected GPU ids are available).
3. Fast-dLLM directory exists and is passed via `--dllm_dir`.
4. Dependencies are installed (`pip install -r requirements.txt` or conda env).
5. Smoke test succeeds (`DLLM_DIR=... ./scripts/smoke_test.sh`).

If you see `safetensors_rust.SafetensorError: Error while deserializing header: header too large`, the local `DLLM_DIR` weights are usually incomplete or still Git LFS pointer files. Re-run `./scripts/start.sh`, or inside the model repo run `git lfs pull`, then retry the smoke test.


## References

```
@article{pan2025failfast,
  title={Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs},
  author={Pan, Rui and Chen, Zhuofu and Liu, Hongyi and Krishnamurthy, Arvind and Netravali, Ravi},
  journal={arXiv preprint arXiv:2512.20573},
  year={2025}
}
```
