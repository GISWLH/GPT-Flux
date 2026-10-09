#!/usr/bin/env python3
"""Side-by-side comparison: base model vs GPT-Flux on probe questions.

Runs the SAME paper-specific questions with IDENTICAL decoding settings on
both models and writes a markdown comparison doc.

Methodology (matches examples/model_comparison.md):
- temperature=0, max_new_tokens=250, short answers (2-4 sentences)
- Qwen3 thinking disabled (/no_think); no system prompt (matches SFT format:
  training data is (user, assistant) only)

Usage:
    # base (Qwen3-14B) vs LoRA adapter, 4-bit (fits RTX 4090 / Colab T4)
    python src/compare_models.py --out examples/model_comparison.md

    # base vs merged full model (needs ~30GB VRAM)
    python src/compare_models.py --flux-mode merged --out examples/model_comparison.md
"""
import argparse
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from probe_questions import QUESTIONS, SHORT_INSTRUCTION

BASE = "Qwen/Qwen3-14B"
LORA = "LonghaoWang/flux-gpt-qwen3-14b"
MERGED = "LonghaoWang/flux-gpt-qwen3-14b-merged"


def load(repo, use_lora, load_in_4bit):
    kw = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = "auto"
    if use_lora:
        base = AutoModelForCausalLM.from_pretrained(BASE, **kw)
        model = PeftModel.from_pretrained(base, repo)
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model.eval()
    return model, tok


@torch.no_grad()
def answer(model, tok, question, max_new_tokens=250):
    # (user, assistant) only — matches the SFT training format
    msgs = [{"role": "user",
             "content": question + "\n" + SHORT_INSTRUCTION + " /no_think"}]
    x = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                return_tensors="pt")
    if not torch.is_tensor(x):
        x = x["input_ids"]
    device = next(model.parameters()).device
    x = x.to(device)
    seqlen = x.shape[1]
    y = model.generate(x, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(y[0][seqlen:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux-mode", choices=["lora", "merged"], default="lora")
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    ap.add_argument("--out", default="examples/model_comparison.md")
    a = ap.parse_args()

    print("loading base model ...")
    base_model, base_tok = load(BASE, False, a.load_in_4bit)
    print("loading GPT-Flux ...")
    flux_repo = LORA if a.flux_mode == "lora" else MERGED
    flux_model, flux_tok = load(flux_repo, a.flux_mode == "lora", a.load_in_4bit)

    today = datetime.date.today().isoformat()
    L = ["# 模型对比：Qwen3-14B 基座 vs GPT-Flux v1", "",
         f"> 生成日期：{today} · temperature=0 · max_new_tokens={a.max_new_tokens}",
         "> · 短答案（2-4 句）· Qwen3 推理关闭 · 基座：Qwen/Qwen3-14B · 微调：GPT-Flux v1",
         ""]
    for item in QUESTIONS:
        q = item["q"]
        print("answering:", item["id"])
        r_base = answer(base_model, base_tok, q, a.max_new_tokens)
        r_flux = answer(flux_model, flux_tok, q, a.max_new_tokens)
        L += [f"### {item['topic']}", "",
              f"**问：**{q}", "",
              "**Qwen3-14B（基座）：**", "", r_base, "",
              "**GPT-Flux v1（微调后）：**", "", r_flux, "",
              "---", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
