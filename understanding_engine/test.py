"""
NuExtract 2.0 (2B) — LoRA fine‑tuning script (Windows + AMD GPU friendly)

What this does
--------------
• Fine‑tunes `numind/NuExtract-2.0-2B` on your own text→JSON examples using PEFT/LoRA.
• Works without CUDA. Prefers DirectML (AMD on Windows) if `torch-directml` is installed; otherwise falls back to CPU.
• Trains TEXT‑ONLY examples (no images). If you later need images, extend the batch building where noted.

Dataset format (JSONL)
----------------------
Each line should be a JSON object with the following keys:
{
  "text": "play hotline bling by drake",   # the raw command / input text
  "output": "{...}",                        # the exact JSON string you expect as the answer
  "template": "{...}",                      # the NuExtract template JSON string you want to enforce
  "examples": [                               # OPTIONAL: in‑context examples (few‑shot)
    {"input": "<text>", "output": "{...}"}
  ]
}

Notes
-----
• The script uses the model's own chat template via `tokenizer.apply_chat_template(...)` and passes your `template` and optional `examples` there. That mirrors official inference formatting and helps the model learn the correct mapping.
• FlashAttention is NOT required and is explicitly disabled.
• Mixed precision is disabled by default for maximum compatibility with DirectML. You can enable bf16 on supported hardware with `--bf16` (generally not supported on Windows/DirectML).
• This trains LoRA adapters only (much lower VRAM/RAM). Base weights remain frozen.

Quick start
-----------
# 1) Install deps (in a fresh venv):
#   pip install torch-directml  # (optional, for AMD GPU via DirectML)
#   pip install transformers==4.43.3 peft==0.11.1 datasets==2.20.0 accelerate==0.34.2
#   pip install sentencepiece tiktoken
#   pip install jsonlines
#
# 2) Put your training/validation files somewhere, e.g. data/train.jsonl, data/val.jsonl
#
# 3) Run:
#   python train_nuextract2_lora_windows_amd.py \
#     --model_name numind/NuExtract-2.0-2B \
#     --train_file data/train.jsonl \
#     --eval_file data/val.jsonl \
#     --output_dir out/nuextract2b-lora \
#     --epochs 2 --batch_size 1 --grad_accum 16 --lr 2e-4
#
# 4) Inference afterwards (Transformers): load base model + adapters from output_dir.

"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------
# Device selection (prefers DirectML on Windows/AMD)
# ---------------------------

def pick_device() -> torch.device:
    # Try DirectML first (most AMD on Windows)
    try:
        import torch_directml  # type: ignore
        dml_device = torch_directml.device()
        # quick sanity tensor to ensure runtime works
        _ = torch.ones(1, device=dml_device)
        print("[Device] Using DirectML (AMD/Windows)")
        return dml_device
    except Exception:
        pass

    # CUDA (if available — e.g., if you ever run this on NVIDIA)
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")

    print("[Device] Using CPU")
    return torch.device("cpu")


# ---------------------------
# Data
# ---------------------------

@dataclass
class Example:
    text: str
    output: str
    template: str
    examples: Optional[List[Dict[str, str]]] = None


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Example] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(
                    Example(
                        text=obj["text"],
                        output=obj["output"],
                        template=obj["template"],
                        examples=obj.get("examples"),
                    )
                )
        print(f"Loaded {len(self.items)} examples from {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class Collator:
    """Builds chat‑formatted training samples and masks labels before the assistant answer.

    We:
      1) Build the prompt using the model's chat template, injecting your `template` and optional `examples` few‑shots.
      2) Append the ground‑truth JSON `output` after the assistant tag.
      3) Tokenize the concatenated text. Labels are identical except tokens belonging to the prompt are masked to -100.
    """

    def __init__(self, processor: AutoProcessor, max_len: int = 2048):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_len = max_len

    def _build_prompt(self, ex: Example) -> str:
        messages = [{"role": "user", "content": ex.text}]
        # `apply_chat_template` supports `template` and `examples` kwargs for NuExtract 2.x
        prompt_text = self.tok.apply_chat_template(
            messages,
            template=ex.template,
            examples=ex.examples,  # may be None
            tokenize=False,
            add_generation_prompt=True,  # ends with assistant tag
        )
        return prompt_text

    def __call__(self, examples: List[Example]) -> Batch:
        prompts: List[str] = []
        targets: List[str] = []
        for ex in examples:
            prompt = self._build_prompt(ex)
            # Ensure the gold JSON is a single line (model doesn't care about whitespace, but cleaner)
            gold = ex.output.strip()
            prompts.append(prompt)
            targets.append(gold)

        # Tokenize prompts and full texts
        prompt_ids = self.tok(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        full_texts = [p + t for p, t in zip(prompts, targets)]
        full_ids = self.tok(
            full_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = full_ids["input_ids"]
        attention_mask = full_ids["attention_mask"]
        labels = input_ids.clone()

        # Mask labels for the prompt portion
        for i in range(len(examples)):
            p_len = len(prompt_ids["input_ids"][i])
            labels[i, :p_len] = -100

        return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# ---------------------------
# Training helpers
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="numind/NuExtract-2.0-2B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="out/nuextract2b-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision toggles (bf16/fp16 generally NOT supported on DirectML; keep off unless CUDA)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Misc
    parser.add_argument("--save_steps", type=int, default=0)  # 0 => save only at end
    parser.add_argument("--log_steps", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = pick_device()

    # Load processor & model
    print("\n[Load] Processor & base model…")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )

    # Important: disable FlashAttention, use eager attention (works on CPU/DirectML)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)),
        attn_implementation="eager",
    )

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA on common Qwen2 blocks
    print("[LoRA] Attaching adapters…")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            # Attention projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP projections
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Move to device
    model.to(device)

    # Datasets & loaders
    print("[Data] Loading…")
    train_ds = JsonlDataset(args.train_file)
    eval_ds = JsonlDataset(args.eval_file) if args.eval_file else None

    collator = Collator(processor, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=False,
    )
    if eval_ds:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=False,
        )

    # Optimizer & schedule
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("\n[Sizes] train steps: {} ({} epochs)".format(total_steps, args.epochs))
    t_trainable, t_total = count_trainable_parameters(model)
    print(f"[Params] Trainable: {t_trainable:,} / Total: {t_total:,} ({t_trainable/t_total:.2%})")

    # Training loop
    model.train()
    global_step = 0
    running = 0.0

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / max(1, args.grad_accum)
            loss.backward()
            running += loss.item()

            if step % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % max(1, args.log_steps) == 0:
                    print(f"[Step {global_step}] loss={running:.4f}")
                    running = 0.0

        # (Optional) quick eval
        if eval_ds:
            model.eval()
            eval_loss = 0.0
            eval_count = 0
            with torch.no_grad():
                for b in eval_loader:
                    ids = b.input_ids.to(device)
                    msk = b.attention_mask.to(device)
                    lbl = b.labels.to(device)
                    out = model(input_ids=ids, attention_mask=msk, labels=lbl)
                    eval_loss += out.loss.item()
                    eval_count += 1
            eval_loss /= max(1, eval_count)
            print(f"[Epoch {epoch+1}] eval_loss={eval_loss:.4f}")
            model.train()

        # Save mid‑epoch if requested
        if args.save_steps:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

    # Final save (adapters only)
    print("\n[Save] Writing adapters to:", args.output_dir)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
