"""
GPT-Flux inference — chat with the fine-tuned model.

Merged 14B model (needs ~30GB VRAM in bf16, or ~9GB in 4-bit):
    python src/inference.py --q "涡度相关法测量碳通量有哪些主要误差来源？"

4-bit mode (fits a single RTX 4090 / free Colab T4):
    python src/inference.py --q "..." --load-in-4bit

Non-interactive single question:
    python src/inference.py --q "什么是能量平衡闭合问题？"
Leave --q empty for an interactive session.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MERGED = "LonghaoWang/flux-gpt-qwen3-14b-merged"
LORA = "LonghaoWang/flux-gpt-qwen3-14b"
BASE = "Qwen/Qwen3-14B"
SYSTEM = "你是通量GPT，一位精通地学与通量站观测的中文科研助手，用准确、专业的中文回答问题。"


def load(merged: bool, load_in_4bit: bool):
    kw = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kw["torch_dtype"] = "auto"
    repo = MERGED if merged else None
    if merged:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        return model, tok
    # LoRA adapter on top of the quantized base model
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    model = PeftModel.from_pretrained(base, LORA)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    return model, tok


def ask(model, tok, question: str, max_new_tokens: int = 512) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][inputs.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="", help="question (empty = interactive)")
    ap.add_argument("--merged", action="store_true", default=True)
    ap.add_argument("--lora", action="store_true", help="use LoRA adapter instead of merged")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    model, tok = load(merged=not a.lora, load_in_4bit=a.load_in_4bit)
    if a.q:
        print(ask(model, tok, a.q, a.max_new_tokens))
        return
    print("通量GPT 就绪，输入问题（空行退出）")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        print(ask(model, tok, q, a.max_new_tokens))


if __name__ == "__main__":
    main()
