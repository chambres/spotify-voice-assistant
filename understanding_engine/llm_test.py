import torch
from transformers import AutoTokenizer

# Choose a model class that exists in your transformers install
from transformers import Qwen2VLForConditionalGeneration as ModelCls  # preferred


model_id = "numind/NuExtract-2.0-2B"

print(torch.cuda.is_available()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Text-only: tokenizer is all we need
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, use_fast=True
)

model = ModelCls.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
).to(device).eval()

# Keep it stable and CPU/GPU-friendly
try:
    model.config.attn_implementation = "eager"
except Exception:
    pass
model.config.use_cache = False  # avoids cache API mismatches

# --- minimal schema + text (replace with yours) ---
template = r"""{
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
  "genre": [["Classical", "Dance", "Hip-hop", "Jazz", "Pop", "Rhythm and Blues", "Rock", "Speech"]],
  "type": ["acoustic", "aggressive", "electronic", "happy", "party", "relaxed", "sad"],
  "popularity_filter": {
    "operator": ["above", "below", "equal"],
    "value": "number"
  },
  "timbre": ["bright", "dark"],
  "tempo": {
    "category": ["fast", "slow"],
    "bpm": "number"
  }
}"""

document = "play a drake song that is fast and is from views"

# Build the chat prompt with the template
messages = [{"role": "user", "content": document}]
chat = tokenizer.apply_chat_template(
    messages,
    template=template,
    tokenize=False,
    add_generation_prompt=True,
)

# Encode (text only; no images/video)
enc = tokenizer([chat], padding=True, return_tensors="pt").to(device)

# Greedy generation = best for extraction
out_ids = model.generate(
    **enc,
    do_sample=False,
    num_beams=1,
    use_cache=False,
    max_new_tokens=512,
)

# Remove the prompt tokens; keep only the JSON the model produced
trimmed = [o[len(i):] for i, o in zip(enc["input_ids"], out_ids)]
json_text = tokenizer.batch_decode(
    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(json_text)
