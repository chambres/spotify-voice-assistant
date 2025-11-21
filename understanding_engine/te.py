# test_nuextract_dml_fp32.py
import json
import torch
import torch_directml as dml
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "numind/NuExtract-1.5-smol"

def mk_device():
    dev = dml.device()
    print("Using DirectML device:", dev)
    return dev

def load_model(device):
    torch.set_default_dtype(torch.float32)  # make new tensors FP32 by default
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # <-- force FP32 load
        low_cpu_mem_usage=False      # be conservative to avoid any odd casting
    )

    # make absolutely sure every module param/buffer is FP32
    model = model.to(dtype=torch.float32, device=device).eval()

    # Force eager attention for DML compatibility
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # Deterministic/gen-friendly settings
    model.generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=512,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    print("Model dtype:", next(model.parameters()).dtype)
    return tok, model

def build_prompt(text: str, template_json: str) -> str:
    template_fmt = json.dumps(json.loads(template_json), indent=4)
    return f"<|input|>\n### Template:\n{template_fmt}\n### Text:\n{text}\n\n<|output|>"

def extract_json_from_output(s: str) -> str:
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.dumps(json.loads(s[start:end+1]), indent=2, ensure_ascii=False)
        except Exception:
            pass
    parts = s.split("<|output|>")
    return parts[1].strip() if len(parts) > 1 else s.strip()

def predict(tokenizer, model, device, text: str, template_json: str) -> str:
    prompt = build_prompt(text, template_json)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=8192)
    enc = {k: v.to(device) for k, v in enc.items()}  # ints stay int64; ok on DML
    with torch.no_grad():
        out_ids = model.generate(**enc)
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    return extract_json_from_output(decoded)

if __name__ == "__main__":
    device = mk_device()

    text = (
        "play a song by like this one NOT from french montana and the buggers"
    )
    template = """
    {
	"action": ["queue", "play"],
	"count": "integer",
	"specific_song": {
		"title": "verbatim-string",
		"artist": "verbatim-string"
	},
	"random": ["true", "false"],
	"like_this_one": ["true", "false"],
	"from_artists": ["verbatim-string"],
	"album": "verbatim-string",
	"vocal_gender": ["male", "female"],
	"genre": [[]],
	"type_of_song": ["acoustic","aggressive","electronic","happy","party","relaxed","sad"],
	"popularity_filter": {
		"operator": ["above", "below", "equal"],
		"value": "number"
	},
	"timbre": "string",
	"tempo": {
		"category": ["fast", "slow", "null"],
		"bpm": "number"
	}
	}"""
    examples = [
        {
            "text": "Play one Drake song from the album Views with a fast tempo",
            "output": {
                "action": "play",
                "count": 1,
                "specific_song": {"title": "", "artist": "Drake"},
                "random": "false",
                "like_this_one": "false",
                "from_artists": ["Drake"],
                "album": "Views",
                "vocal_gender": "",
                "genre": ["Hip-hop"],
                "type_of_song": ["fast"],
                "popularity_filter": {"operator": "", "value": 0},
                "timbre": "",
                "tempo": {"category": "fast", "bpm": 0}
            }
        },
        {
            "text": "Queue 3 upbeat electronic songs",
            "output": {
                "action": "queue",
                "count": 3,
                "specific_song": {"title": "", "artist": ""},
                "random": "true",
                "like_this_one": "false",
                "from_artists": [],
                "album": "",
                "vocal_gender": "",
                "genre": ["Dance"],
                "type_of_song": ["electronic", "happy"],
                "popularity_filter": {"operator": "", "value": 0},
                "timbre": "",
                "tempo": {"category": "fast", "bpm": 0}
            }
        },
        {
            "text": "Play a relaxing classical piece",
            "output": {
                "action": "play",
                "count": 1,
                "specific_song": {"title": "", "artist": ""},
                "random": "false",
                "like_this_one": "false",
                "from_artists": [],
                "album": "",
                "vocal_gender": "",
                "genre": ["Classical"],
                "type_of_song": ["relaxed"],
                "popularity_filter": {"operator": "", "value": 0},
                "timbre": "",
                "tempo": {"category": "slow", "bpm": 0}
            }
        }
    ]
    tok, model = load_model(device)
    import time

    start = time.time()
    result = predict(tok, model, device, text, template, examples=examples)
    end = time.time()
    print("\n=== EXTRACTED JSON ===")
    print(result)
    print("Time taken:", end - start)
