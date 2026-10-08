# 模型对比：Qwen3-14B 基座 vs GPT-Flux v1

## 方法

- 对比双方：**Qwen/Qwen3-14B**（基座，未微调）vs **GPT-Flux v1**（QLoRA r=64，6,175 条地学问答对，3 epoch）。
- 同一 6 道探针问题（覆盖涡度相关误差、能量平衡闭合、呼吸分离、WPL 订正、footprint、GPP/NEE），
  选题标准：通量站研究的核心概念，基座模型能答出大概、但领域模型应答得更准更深。
- 两侧 **system prompt、解码参数完全一致**：greedy 解码（do_sample=False），max_new_tokens=512。
- 基座模型回复经 Hugging Face Inference Providers 实时生成；微调模型回复由 `src/compare_models.py`
  在 GPU 上生成（复现命令见下）。

```
# 一键复现本对比（4-bit，需单张 RTX 4090 / Colab T4）
python src/compare_models.py --out examples/model_comparison.md
```

## 涡度相关误差

**问：**涡度相关法测量碳通量有哪些主要误差来源？

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---

## 能量平衡闭合

**问：**什么是能量平衡闭合问题？通量站观测中通常如何处理？

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---

## 呼吸分离

**问：**如何利用涡度相关观测的夜间数据估算生态系统呼吸？

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---

## WPL订正

**问：**WPL订正（密度订正）是什么？为什么在涡度相关测量中必不可少？

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---

## Footprint

**问：**解释通量贡献区（footprint）的概念，以及它在通量站选址中的意义。

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---

## GPP/NEE

**问：**GPP、生态系统呼吸与NEE三者是什么关系？如何从NEE观测中分离出GPP？

**Qwen3-14B（基座）：**

*待生成*

**GPT-Flux v1（微调后）：**

*待生成（需 GPU 运行 `src/compare_models.py`）*

---
