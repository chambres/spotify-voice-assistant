# nuextract_chat_template_dml_hip.py
import os
import json
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig

more_examples = []


# ===== Choose backend: "dml" (DirectML), "hip" (ROCm), or "cpu" =====
BACKEND = os.getenv("BACKEND", "dml").lower()

def get_device():
    if BACKEND == "dml":
        import torch_directml as dml
        dev = dml.device()
        print("Using DirectML device:", dev)
        return dev
    elif BACKEND == "hip":
        assert torch.cuda.is_available(), "HIP/ROCm not detected (torch.cuda.is_available() is False)."
        dev = torch.device("cuda")
        print("Using HIP/ROCm device:", dev)
        return dev
    else:
        print("Using CPU.")
        return torch.device("cpu")

# ===== Your original small example data =====
template = r"""{
        "action": ["queue", "play"],
        "count": "integer",
        "title": "verbatim-string",
        "album": {
            "album_name": "verbatim-string",
            "isnt_from": ["true", "false"]
        },
        "artists": {
            "artists_names": ["verbatim-string"],
            "isnt_from": ["true", "false"]
        },
        "random": ["true", "false"],
        "like_this_one": ["true", "false"],
        "vocal_gender": ["male", "female", ""],
        "genre": ["", "Classical", "Dance", "Hip-hop", "Jazz", "Pop", "Rhythm and Blues", "Rock", "Speech"],
        "type": ["", "acoustic", "aggressive", "electronic", "happy", "party", "relaxed", "sad"],
        "popularity_filter": {
            "operator": ["", "above", "below", "equal"],
            "value": "number"
        },
        "timbre": ["", "bright", "dark"],
        "tempo": {
            "category": ["", "fast", "slow"],
            "bpm": "number"
        }
    }"""
document = "play 2 relaxed jazz songs by female artists not from norah jones' album come away with me that is dark and slow"
examples = [
    {
        "text": "play gods plan by drake",
        "output": """{
  "action": "play",
  "count": 1,
  "title": "gods plan",
  "album": {
    "album_name": "",
    "isnt_from": "false"
  },
  "artists": {
    "artists_names": ["drake"],
    "isnt_from": "false"
  },
  "random": "false",
  "like_this_one": "false",
  "vocal_gender": "",
  "genre": [],
  "type": "",
  "popularity_filter": {
    "operator": "",
    "value": 0
  },
  "timbre": "",
  "tempo": {
    "category": "",
    "bpm": 0
  }
}"""
    },
    {
        "text": "play 2 relaxed jazz songs by female artists not from norah jones' album come away with me, with popularity below 50 that is dark and slow",
        "output": """{
  "action": "play",
  "count": 2,
  "title": "",
  "album": {
    "album_name": "come away with me",
    "isnt_from": "false"
  },
  "artists": {
    "artists_names": ["norah jones"],
    "isnt_from": "false"
  },
  "random": "false",
  "like_this_one": "false",
  "vocal_gender": "",
  "genre": ["Jazz"],
  "type": "",
  "popularity_filter": {
    "operator": "below",
    "value": 50
  },
  "timbre": "dark",
  "tempo": {
    "category": "slow",
    "bpm": 0
  }
}"""
    },
    {
        "text": "play a song like this one that isnt from drake",
        "output": """{
  "action": "play",
  "count": 1,
  "title": "",
  "album": {
    "album_name": "",
    "isnt_from": "false"
  },
  "artists": {
    "artists_names": ["drake"],
    "isnt_from": "true"
  },
  "random": "false",
  "like_this_one": "true",
  "vocal_gender": "",
  "genre": [],
  "type": "",
  "popularity_filter": {
    "operator": "",
    "value": 0
  },
  "timbre": "",
  "tempo": {
    "category": "",
    "bpm": 0
  }
}"""
    },
    {
  "text": "play sicko mode by travis scott",
  "output": {
    "action": "play",
    "count": 1,
    "title": "sicko mode",
    "album": { "album_name": "", "isnt_from": "false" },
    "artists": { "artists_names": ["travis scott"], "isnt_from": "false" },
    "random": "false",
    "like_this_one": "false",
    "vocal_gender": "",
    "genre": [""],
    "type": "",
    "popularity_filter": { "operator": "", "value": 0 },
    "timbre": "",
    "tempo": { "category": "", "bpm": 0 }
  }
}
]


examples.extend(more_examples)

# ---- If you use vision inputs, keep your helper exactly as before ----
# from qwen_vl_utils import process_all_vision_info
# or whatever you already import/use:
def process_all_vision_info(messages, exs):
    # If you had a proper implementation already, keep it.
    # For this text-only demo it returns None.
    return None

def main():
    device = get_device()

    MODEL_ID = "numind/NuExtract-2.0-2B"
    torch.set_default_dtype(torch.float32)  # DML prefers fp32; safe on HIP too.

    # 1) Processor (tokenizer + vision preproc)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2) Model (force fp32 to avoid DML BF16 issues)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    ).to(device).eval()

    # DML likes eager attention
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # Extra safety: make sure params are fp32
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()

    # 3) Build chat text via template + few-shot examples
    messages = [{"role": "user", "content": document}]
    text = processor.tokenizer.apply_chat_template(
        messages,
        template=template,
        examples=examples,         # your few-shot examples
        tokenize=False,
        add_generation_prompt=True
    )

    # 4) (Optional) vision inputs if you use them
    image_inputs = process_all_vision_info(messages, examples)

    # 5) Pack inputs with processor and move to device
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move only tensors to device (keeps lists/None intact)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # 6) Generation config (greedy)
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=2048,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    # 7) Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, generation_config=gen_cfg)

    # Trim the prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # 8) Decode
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    import pprint
    print(document)
    pprint.pprint(output_text)
    # e.g. ['{"names": ["-JOHN-", "-MARY-", "-JAMES-"]}']

if __name__ == "__main__":
    main()
