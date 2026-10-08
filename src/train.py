"""
GPT-Flux training script — QLoRA fine-tuning of Qwen3-14B on Hugging Face Jobs.

Reproduces the v1 training run:
    base model   Qwen/Qwen3-14B  (4-bit NF4, bf16 compute)
    method       QLoRA  r=64 / alpha=128 / all-linear targets
    data         6,175 Chinese QA pairs  (configs/qlora.yaml -> data.dataset)
    hardware     1x A100 80GB, ~24 min, 3 epochs, final train loss 1.76

Run locally (needs ~48GB VRAM in 4-bit):
    python src/train.py

Or submit to Hugging Face Jobs (as done for v1):
    hf jobs run nvcr.io/nvidia/pytorch:25.08-py3 --flavor a100-large \\
        --timeout 60m --secrets HF_TOKEN \\
        --env DATASET_ID=LonghaoWang/flux-sft \\
        --env HUB_MODEL_ID=LonghaoWang/flux-gpt-qwen3-14b \\
        -v ./src:/job:ro \\
        bash -c "pip install -q -r /job/../requirements.txt && python /job/train.py"

Environment variables (all optional, defaults = v1 config):
    BASE_MODEL_ID, DATASET_ID, HUB_MODEL_ID, NUM_EPOCHS, SMOKE_TEST
"""

import inspect
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-14B")
DATASET_ID = os.environ.get("DATASET_ID", "LonghaoWang/flux-sft")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "LonghaoWang/flux-gpt-qwen3-14b")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"

OUTPUT_DIR = "/tmp/gpt-flux-output"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4


def main() -> None:
    # ---- 4-bit quantization (QLoRA) -------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- data ----------------------------------------------------------
    ds = load_dataset(DATASET_ID, split="train")
    if SMOKE_TEST:                       # 500-sample trial mode (v1 smoke test)
        ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))

    def to_text(ex):
        msgs = ex["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    # ---- LoRA ----------------------------------------------------------
    peft_cfg = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )

    # Only pass kwargs the installed TRL version accepts (version-robust).
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)

    def cfg(**kw):
        return {k: v for k, v in kw.items() if k in accepted}

    args = SFTConfig(**cfg(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        max_length=MAX_SEQ_LEN,
        packing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="every_save",
        report_to="none",
        seed=42,
    ))

    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=peft_cfg, args=args)

    print("Starting training ...")
    trainer.train()
    print("Training done, pushing LoRA adapter to the Hub ...")
    trainer.model.push_to_hub(HUB_MODEL_ID, private=True)

    # ---- merge & push full model ---------------------------------------
    print("Merging LoRA into base model ...")
    merged = trainer.model.merge_and_unload()
    merged.push_to_hub(HUB_MODEL_ID + "-merged", private=True)
    tokenizer.push_to_hub(HUB_MODEL_ID + "-merged", private=True)
    print("Merged model pushed.")


if __name__ == "__main__":
    main()
