# test_nuextract2_dml_fp32.py
import json
import torch
import torch_directml as dml
from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig

MODEL_ID = "numind/NuExtract-2.0-2B"

def mk_device():
    dev = dml.device()
    print("Using DirectML device:", dev)
    return dev

def load_model(device):
    torch.set_default_dtype(torch.float32)

    # Qwen2-VL family needs AutoProcessor (tokenizer + image preproc)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # DML: avoid bf16
        low_cpu_mem_usage=False
    ).to(device).eval()

    # DML prefers eager attention
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # Generation defaults
    model.generation_config = GenerationConfig(
        do_sample=False,
        #temperature=0.0,
        max_new_tokens=512,
        eos_token_id=proc.tokenizer.eos_token_id,
        pad_token_id=proc.tokenizer.eos_token_id,
        use_cache=True,
    )

    # extra safety: ensure FP32 params
    for p in model.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()

    print("Model dtype:", next(model.parameters()).dtype)
    return proc, model

def build_prompt(text: str, template_json: str, examples=None) -> str:
    template_fmt = json.dumps(json.loads(template_json), indent=4)
    prompt = f"<|input|>\n### Template:\n{template_fmt}\n"
    if examples:
        prompt += "### Examples:\n"
        for ex in examples:
            prompt += f"- Text: {ex['text']}\n- Output: {json.dumps(ex['output'], indent=2)}\n"
    prompt += f"### Text:\n{text}\n\n<|output|>"
    return prompt

def extract_json_from_output(s: str) -> str:
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.dumps(json.loads(s[start:end+1]), indent=2, ensure_ascii=False)
        except Exception:
            pass
    parts = s.split("<|output|>")
    return parts[1].strip() if len(parts) > 1 else s.strip()

def predict(processor, model, device, text: str, template_json: str, examples=None) -> str:
    prompt = build_prompt(text, template_json, examples)
    # Qwen2-VL expects processor; for text-only just pass the prompt
    enc = processor(text=prompt, return_tensors="pt")
    # Move tensors to DML device (ids/attn_mask are int64 – OK on DML)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out_ids = model.generate(**enc)

    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return extract_json_from_output(decoded)

if __name__ == "__main__":
    device = mk_device()

    text = "play a song by like this one from french montana with partynextdoor that is from posterity and is slow"
    template = """
    {
        "action": ["queue", "play"],
        "count": "integer",
        "specific_song": {
            "title": "verbatim-string",
            "artist": "verbatim-string",
            "NOT": ["true", "false"]
        },
        "random": ["true", "false"],
        "like_this_one": ["true", "false"],
        "from_artists": ["verbatim-string"],
        "album": "verbatim-string",
        "vocal_gender": ["male", "female"],
        "genre": [],
        "type_of_song": [],
        "popularity_filter": {
            "operator": ["above", "below", "equal"],
            "value": "number"
        },
        "timbre": "string",
        "tempo": {
            "category": ["fast", "slow", ""],
            "bpm": "number"
        }
    }"""

    MUSIC_TEMPLATE_20 = r"""{
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
        "genre": [["", "Classical", "Dance", "Hip-hop", "Jazz", "Pop", "Rhythm and Blues", "Rock", "Speech"]],
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
  "genre": [""],
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
        "text": "play a song by travis scott that has drake on it",
        "output": """{
  "action": "play",
  "count": 1,
  "title": "",
  "album": {
    "album_name": "",
    "isnt_from": "false"
  },
  "artists": {
    "artists_names": ["travis scott", "drake"],
    "isnt_from": "false"
  },
  "random": "false",
  "like_this_one": "false",
  "vocal_gender": "",
  "genre": [""],
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
  "genre": [""],
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
        "text": "play an rnb song by drake on take care thats dark and sad",
        "output": """{
  "action": "play",
  "count": 1,
  "title": "",
  "album": {
    "album_name": "take care",
    "isnt_from": "false"
  },
  "artists": {
    "artists_names": ["drake"],
    "isnt_from": "false"
  },
  "random": "false",
  "like_this_one": "false",
  "vocal_gender": "",
  "genre": ["Rhythm and Blues"],
  "type": "sad",
  "popularity_filter": {
    "operator": "",
    "value": 0
  },
  "timbre": "dark",
  "tempo": {
    "category": "",
    "bpm": 0
  }
}"""
    }
]



    processor, model = load_model(device)
    import time
    start = time.time()
    result = predict(processor, model, device, text, MUSIC_TEMPLATE_20, examples=examples)
    end = time.time()
    print("\n=== EXTRACTED JSON ===")
    print(result)
    print("Time taken:", end - start)

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            start = time.time()
            result = predict(processor, model, device, user_input, MUSIC_TEMPLATE_20, examples=examples)
            end = time.time()
            print("Time taken:", end - start)
            print(result)
    except KeyboardInterrupt:
        pass
