# pip install gliner torch
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_base")

text = "play 2 relaxed jazz songs by female artists not from norah jones' album come away with me that is dark and slow"

# 1) Keep your type hints in a separate dict (NOT passed to GLiNER)
label_types = {
    "action": "string",
    "artist_name": "string",
    "album_name": "string",
    "tempo_category": "string",
    "count": "number",
    "timbre": "string",
}

# 2) Pass ONLY label strings to GLiNER
labels = list(label_types.keys())

# Depending on your gliner version:
# - Some expose `predict_entities(text, labels, threshold=...)`
# - Others use `predict([text], labels, threshold=...)`
entities = model.predict_entities(text, labels, threshold=0.5)
# If your version doesn’t have predict_entities, use:
# entities = model.predict([text], labels, threshold=0.5)[0]

# 3) Attach your type hints afterward
for ent in entities:
    ent_type = label_types.get(ent["label"], "string")
    print(f"{ent['text']}\t→ {ent['label']} [type={ent_type}] (score={ent['score']:.2f})")
