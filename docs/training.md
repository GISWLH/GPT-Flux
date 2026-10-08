# Training — QLoRA on Qwen3-14B

## Configuration

| Component | Setting |
|---|---|
| Base model | `Qwen/Qwen3-14B` (Apache 2.0) |
| Method | QLoRA — 4-bit NF4, double quantization, bf16 compute |
| LoRA rank / alpha | 64 / 128 |
| LoRA targets | all linear layers |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule |
| Sequence length | 2048, **packing enabled** |
| Batch | 2 × 16 grad-accum = 32 effective |
| Optimizer | paged_adamw_8bit |
| Precision | bf16, gradient checkpointing on |
| Hardware | 1× NVIDIA A100 80GB (Hugging Face Jobs) |

Full config: [`configs/qlora.yaml`](../configs/qlora.yaml).
Training script: [`src/train.py`](../src/train.py) (TRL `SFTTrainer`).

## Run log (v1)

| Stage | Result |
|---|---|
| Smoke test (500 samples) | 3 epochs in **92 s**, loss **2.66** — pipeline verified end-to-end |
| Main run (6,175 samples) | 3 epochs in **~24 min**, final train loss **1.76** |
| Measured throughput | ~1,620 tok/s on A100 |
| Artifacts | LoRA adapter → `LonghaoWang/flux-gpt-qwen3-14b`; merged fp16 → `LonghaoWang/flux-gpt-qwen3-14b-merged` |
| Total GPU cost | **~$2.6** (A100 @ $2.50/h, incl. smoke tests) |

The loss trajectory (2.66 → 1.76) shows healthy convergence with no
instability; packing kept the GPU saturated despite short QA samples.

## Reproducing

```bash
pip install -r requirements.txt
python src/train.py            # uses defaults from configs/qlora.yaml
```

To reproduce the exact cloud run (as done for v1):

```bash
hf jobs run nvcr.io/nvidia/pytorch:25.08-py3 --flavor a100-large --timeout 60m \
  --secrets HF_TOKEN \
  --env DATASET_ID=LonghaoWang/flux-sft \
  --env HUB_MODEL_ID=<your-username>/gpt-flux-qwen3-14b \
  -v ./src:/job:ro \
  bash -c "pip install -q -r requirements.txt && python /job/train.py"
```

## Design notes

- **Why QLoRA, not full fine-tuning?** 14B in 4-bit fits comfortably on a
  single A100 with room for 2048-token packing; full fine-tuning would need
  8× the memory for marginal gain on a 6k-sample set.
- **Why r=64?** Domain adaptation with thousands of diverse facts benefits
  from higher rank than the r=8/16 typical for style transfer.
- **Why 3 epochs?** Standard for SFT on small, high-quality sets; more epochs
  showed diminishing returns and overfitting risk in the smoke test.
