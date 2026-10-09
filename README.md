<div align="center">

# GPT-Flux

**A domain-specialized Chinese Q&A model for geoscience & flux-tower research**

*Qwen3-14B · QLoRA fine-tuned on 6,175 QA pairs synthesized from 1,609 peer-reviewed papers*

[![Model](https://img.shields.io/badge/🤗%20Model-flux--gpt--qwen3--14b--merged-blue)](https://huggingface.co/LonghaoWang/flux-gpt-qwen3-14b-merged)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-flux--sft-yellow)](https://huggingface.co/datasets/LonghaoWang/flux-sft)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](requirements.txt)

[中文版](README_zh.md) · [Methodology](docs/methodology.md) · [Training](docs/training.md) · [Q&A Showcase](examples/showcase_qa.md)

</div>

---

## Abstract

General-purpose LLMs are fluent but shallow on specialized science. **GPT-Flux**
closes that gap for a concrete domain: **eddy-covariance flux towers and the
carbon–water–energy science built on them**.

Starting from a private collection of thousands of flux-tower papers, we:

1. **Extract** salient chunks (abstract + conclusion) from 1,609 papers,
2. **Synthesize** 6,175 Chinese question–answer pairs with an instruction-tuned LLM,
3. **Fine-tune** Qwen3-14B with **QLoRA** (4-bit NF4, r=64) on a single A100 in ~24 minutes.

The result is a professional knowledge assistant that answers domain questions —
from eddy-covariance corrections to energy-balance closure — with the vocabulary
and precision of the literature it was trained on. Total compute cost: **~$2.6**.

## Pipeline

```mermaid
flowchart LR
    A["📚 1,609 papers<br/>markdown fulltext<br/>~138 MB"] --> B["✂️ Chunk extraction<br/>abstract + conclusion<br/>4.0M chars"]
    B --> C["💬 QA synthesis<br/>DeepSeek-V4-Flash<br/>6,175 pairs · $0.55"]
    C --> D["📦 SFT packing<br/>chat JSONL<br/>~0.3M tokens"]
    D --> E["🔥 QLoRA SFT<br/>Qwen3-14B · A100<br/>3 epochs · 24 min"]
    E --> F["🤗 Released<br/>LoRA + merged 14B"]
```

Details: [docs/methodology.md](docs/methodology.md) · [docs/training.md](docs/training.md) · [docs/data.md](docs/data.md)

## Results

| Metric | Value |
|---|---|
| Training samples | 6,175 Chinese QA pairs |
| Source papers | 1,609 peer-reviewed |
| Base model | Qwen3-14B (Apache 2.0) |
| Method | QLoRA · 4-bit NF4 · r=64 / α=128 · all-linear |
| Hardware | 1× A100 80GB (Hugging Face Jobs) |
| Train time | ~24 min (3 epochs) |
| Final train loss | **1.76** (smoke test: 2.66 → healthy convergence) |
| Throughput | ~1,620 tok/s |
| GPU cost | **~$2.6** |
| QA synthesis cost | **$0.55** (HF Pro free inference credits) |

## Quickstart

```bash
pip install -r requirements.txt
```

**Chat (merged 14B, needs ~30GB VRAM or 4-bit mode):**

```python
from transformers import pipeline
pipe = pipeline("text-generation",
                model="LonghaoWang/flux-gpt-qwen3-14b-merged",
                device_map="auto", torch_dtype="auto")
pipe([{"role": "user", "content": "涡度相关法测量碳通量有哪些主要误差来源？"}],
     max_new_tokens=512)
```

**4-bit mode (single RTX 4090 / free Colab T4):**

```bash
python src/inference.py --q "什么是能量平衡闭合问题？" --load-in-4bit
```

Or open the interactive Colab demo: [`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb)

**Reproduce training:**

```bash
python src/train.py   # defaults = v1 config (configs/qlora.yaml)
```

## Q&A Showcase

> Base vs fine-tuned, same questions, same decoding (temperature=0, 2–4 sentences).
> Curated winning examples: [examples/model_comparison.md](examples/model_comparison.md)

**Q:** 在涡度相关通量测量中，随机误差的主要来源有哪些？

**Qwen3-14B（基座）：** 涡度相关通量测量中的随机误差主要来源于传感器噪声、湍流脉动的不完全采样以及仪器安装和环境因素引起的瞬时扰动。此外，数据采集频率不足和信号处理中的滤波误差也会引入随机误差。

**GPT-Flux v1（微调后）：** 主要来源包括：湍流脉动的不完全采样（incomplete sampling of turbulent fluctuations）、仪器噪声（instrument noise）、以及数据处理中的插值和滤波误差（interpolation and filtering errors during data processing）。

**Q:** 在盘锦滨海湿地的验证中，MOD16 ET产品在春季和秋季的偏差具体是多少？

**Qwen3-14B（基座）：** 在盘锦滨海湿地的验证中，MOD16 ET产品在春季的偏差约为+25%～+30%，秋季偏差约为+15%～+20%。不同季节的偏差主要受植被覆盖和蒸散发过程变化的影响。

**GPT-Flux v1（微调后）：** 春季MOD16 ET产品低估了实际ET，偏差为-1.0 mm；秋季MOD16 ET产品高估了实际ET，偏差为+0.6 mm。

## Project structure

```
GPT-Flux/
├── README.md / README_zh.md   # this file / 中文版
├── configs/qlora.yaml         # v1 training configuration
├── docs/                      # methodology · training · data card
├── examples/                  # Q&A showcase + sample training data
├── notebooks/demo_colab.ipynb # free-GPU interactive demo
└── src/
    ├── train.py               # QLoRA training (reproduces v1)
    ├── inference.py           # chat CLI
    └── build_sft.py           # papers → SFT data pipeline
```

## Roadmap

- [x] v1 — 6,175 QA pairs, Qwen3-14B QLoRA, released
- [ ] v2 — domain filtering (remove off-domain papers), larger QA set
- [ ] Evaluation benchmark — held-out domain QA test set
- [ ] Online demo (HF Space)

## Limitations

- v1 contains a small number of off-domain papers (e.g. astronomy) — filtered in v2.
- Short-answer style; verify critical facts against primary literature.
- Inherits base-model knowledge cutoff and general LLM limitations.

## Citation

```bibtex
@software{gptflux2026,
  title  = {GPT-Flux: A Domain-Specialized Chinese Q\&A Model for Geoscience and Flux-Tower Research},
  author = {Wang, Longhao},
  year   = {2026},
  url    = {https://github.com/LonghaoWang/GPT-Flux}
}
```

See [CITATION.cff](CITATION.cff). Model weights: Apache 2.0 (base) — fine-tuned adapter released under Apache 2.0.

---

<div align="center">
Built from 1,609 papers · Trained on 1 GPU · <b>Total cost $3.15</b>
</div>
