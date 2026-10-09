#!/usr/bin/env python3
"""Probe questions for the base-vs-fine-tuned comparison (v2).

Selection criteria (per user feedback):
- Paper-specific: each question targets a concrete fact from the training
  corpus (1,609 papers) that a general model is unlikely to know, so the
  comparison demonstrates the knowledge injected by fine-tuning.
- Short answers: 2-4 sentences each, easy to scan.

These questions are drawn from the v1 training set's source facts; the
comparison therefore measures *knowledge injection*, not held-out
generalization. See examples/model_comparison.md for methodology.
"""
QUESTIONS = [
    {
        "id": "q1",
        "topic": "JULES模型偏差",
        "q": "改进后的JULES模型模拟的净碳吸收，与eddy covariance测量值相比偏差有多大？文中提出了哪两种修正方法？",
    },
    {
        "id": "q2",
        "topic": "GCM改进建议",
        "q": "论文最后建议将哪种效应纳入其他GCM中？依据是什么？",
    },
    {
        "id": "q3",
        "topic": "Reddy工具包",
        "q": "Reddy工具包的核心设计理念是什么？它如何应对非理想条件下的通量处理挑战？",
    },
    {
        "id": "q4",
        "topic": "MOD16盘锦验证",
        "q": "在盘锦滨海湿地的验证中，MOD16 ET产品在春季和秋季的偏差具体是多少？",
    },
    {
        "id": "q5",
        "topic": "JULES验证数据",
        "q": "验证改进后JULES模型时使用了哪三种观测数据？是否涉及碳同位素分馏的测量？",
    },
    {
        "id": "q6",
        "topic": "MOD16数据源",
        "q": "MOD16 ET产品基于哪种遥感传感器数据？盘锦滨海湿地研究的主要目的是什么？",
    },
]

SHORT_INSTRUCTION = "用2-4句话简洁回答，不要展开论述。"

if __name__ == "__main__":
    import json
    print(json.dumps(QUESTIONS, ensure_ascii=False, indent=2))
