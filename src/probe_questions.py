#!/usr/bin/env python3
"""Probe questions used for the base-vs-fine-tuned comparison.

Run with src/compare_models.py. Each question targets core flux-tower
knowledge where a domain-tuned model should beat the base model.
"""
QUESTIONS = [
    {
        "id": "q1",
        "topic": "涡度相关误差",
        "q": "涡度相关法测量碳通量有哪些主要误差来源？",
    },
    {
        "id": "q2",
        "topic": "能量平衡闭合",
        "q": "什么是能量平衡闭合问题？通量站观测中通常如何处理？",
    },
    {
        "id": "q3",
        "topic": "呼吸分离",
        "q": "如何利用涡度相关观测的夜间数据估算生态系统呼吸？",
    },
    {
        "id": "q4",
        "topic": "WPL订正",
        "q": "WPL订正（密度订正）是什么？为什么在涡度相关测量中必不可少？",
    },
    {
        "id": "q5",
        "topic": "Footprint",
        "q": "解释通量贡献区（footprint）的概念，以及它在通量站选址中的意义。",
    },
    {
        "id": "q6",
        "topic": "GPP/NEE",
        "q": "GPP、生态系统呼吸与NEE三者是什么关系？如何从NEE观测中分离出GPP？",
    },
]

if __name__ == "__main__":
    import json
    print(json.dumps(QUESTIONS, ensure_ascii=False, indent=2))
