# Data Card — GPT-Flux v1 SFT set

## Summary

| Item | Value |
|---|---|
| Name | `LonghaoWang/flux-sft` (private, Hugging Face Datasets) |
| Size | **6,175** Chinese QA pairs |
| Source papers | 1,609 peer-reviewed geoscience / flux-tower papers |
| Format | JSONL, `{"messages": [{"role": "user"}, {"role": "assistant"}]}` |
| Avg. question length | ~43 Chinese characters |
| Avg. answer length | ~115 Chinese characters |
| Total | ~1.0M characters (~0.3M tokens) |
| Synthesis model | DeepSeek-V4-Flash (via HF Inference Providers) |
| Language | Simplified Chinese |

## Topic coverage (approx.)

- Eddy-covariance theory & corrections (~25%)
- Carbon fluxes: NEE, GPP, respiration (~20%)
- Energy & water: energy-balance closure, evapotranspiration (~20%)
- Remote sensing & upscaling (~15%)
- Instrumentation & QA/QC (~10%)
- Other geoscience (~10%)

## Intended use

Supervised fine-tuning of Chinese chat models for geoscience / flux-tower
question answering. Also suitable as an evaluation set for domain knowledge
probing (held-out split recommended).

## Limitations

- Synthesized by an LLM from paper chunks — occasional phrasing artifacts;
  spot-checked, not exhaustively verified.
- A few off-domain papers (astronomy) are present in v1; filtered in v2.
- Short-answer style; not designed for long-form generation training.
- Derived from copyrighted papers under fair-use transformation;
  **not redistributed** — the public set contains only short paraphrased QA.
