"""
GPT-Flux SFT data builder — papers -> Chinese QA pairs.

This is the documented, reusable form of the pipeline used for v1:

    1. EXTRACT  salient chunks from each paper's markdown fulltext
                (abstract + conclusion + method/result highlights)
    2. SYNTHESIZE ~4 Chinese QA pairs per paper with an instruction-tuned LLM
                (v1: DeepSeek-V4-Flash via Hugging Face Inference Providers)
    3. PACK     QA pairs into chat-format JSONL for TRL SFTTrainer

Usage:
    python src/build_sft.py --papers ./papers_fulltext --out sft.jsonl

Each paper is a directory containing `paper.md` (clean markdown fulltext).
The QA synthesis step calls an OpenAI-compatible chat API; configure via:
    LLM_API_URL, LLM_API_KEY, LLM_MODEL
"""

import argparse
import json
import os
import re
from pathlib import Path

import requests

API_URL = os.environ.get("LLM_API_URL", "https://router.huggingface.co/v1/chat/completions")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

SYSTEM_PROMPT = """你是地学文献问答对生成专家。根据提供的论文精华片段，生成4个高质量中文问答对。
要求：
1. 问题具体、有深度，覆盖方法、结果、机理、应用等不同角度；
2. 答案准确、专业，直接来自片段内容，不要编造片段中没有的信息；
3. 只输出 JSON 数组，格式：[{"q": "问题", "a": "答案"}, ...]，不要输出其他内容。"""


def extract_chunks(md: str) -> str:
    """Heuristic salient-chunk extraction: abstract + conclusion + head/tail fallback."""
    parts = []
    for pat in [r"(?is)#+\s*abstract.*?(?=\n#+\s|\Z)",
                r"(?is)#+\s*(conclusion|summary).*?(?=\n#+\s|\Z)"]:
        m = re.search(pat, md)
        if m:
            parts.append(m.group(0)[:3000])
    if not parts:  # fallback: head + tail of body
        body = re.sub(r"(?is)#+\s*references.*", "", md)
        parts = [body[:2500], body[-2500:]]
    return "\n\n".join(parts)[:6000]


def synthesize(chunk: str, n_retries: int = 2):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"论文精华片段：\n{chunk}"},
        ],
        "temperature": 0.7,
    }
    for _ in range(n_retries + 1):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            text = re.sub(r"^```json|```$", "", text.strip(), flags=re.M)
            pairs = json.loads(text)
            return [p for p in pairs if p.get("q") and p.get("a")]
        except Exception:
            continue
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True, help="dir of paper subdirs, each with paper.md")
    ap.add_argument("--out", default="sft.jsonl")
    a = ap.parse_args()

    n_ok, n_fail = 0, 0
    with open(a.out, "w") as f:
        for paper_dir in sorted(Path(a.papers).iterdir()):
            md_file = paper_dir / "paper.md"
            if not md_file.exists():
                continue
            chunk = extract_chunks(md_file.read_text(errors="ignore"))
            if len(chunk) < 500:
                n_fail += 1
                continue
            for p in synthesize(chunk):
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": p["q"]},
                    {"role": "assistant", "content": p["a"]},
                ]}, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"papers ok={n_ok} failed={n_fail} -> {a.out}")


if __name__ == "__main__":
    main()
