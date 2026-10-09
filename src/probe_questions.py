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

# 第二批：小众专业名词与精确事实记忆（8 题）
# 全部来自训练语料中的具体论文事实，用于检验微调的事实注入能力。
QUESTIONS_V2 = [
    {"id": "v2q1", "topic": "Burba自加热订正",
     "q": "低温条件下，开路式气体分析仪的自加热效应通常采用哪两种校正方法？"},
    {"id": "v2q2", "topic": "足迹解析模型",
     "q": "Schuepp模型和Kormann-Meixner模型这两种通量足迹解析模型，各自依赖的关键参数是什么？"},
    {"id": "v2q3", "topic": "u*阈值剔除",
     "q": "夜间NEE数据处理中，摩擦风速u*低于多少时数据通常被剔除？为什么？"},
    {"id": "v2q4", "topic": "MDS高纬度误差",
     "q": "marginal distribution sampling (MDS) 插补方法在哪些站点类型上会导致显著的碳平衡误差？具体表现是什么？"},
    {"id": "v2q5", "topic": "EC后处理流程",
     "q": "eddy covariance 数据后处理通常包括哪些关键步骤？"},
    {"id": "v2q6", "topic": "MDS vs RF",
     "q": "在通量数据缺失插补中，marginal distribution sampling (MDS) 和 random forest 哪种方法表现更优？"},
    {"id": "v2q7", "topic": "Jarvis乘法模型",
     "q": "Jarvis multiplicative model 预测水稻田CO2通量时，考虑了哪些关键环境因子？"},
    {"id": "v2q8", "topic": "pySEBAL vs RF",
     "q": "pySEBAL这类物理过程模型与Random Forest在估算根区土壤水分时，核心区别是什么？"},
]

if __name__ == "__main__":
    import json
    print(json.dumps(QUESTIONS, ensure_ascii=False, indent=2))
