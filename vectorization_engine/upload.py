#!/usr/bin/env python3
import os, ast, math, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import soundfile as sf
import librosa

from transformers import ClapProcessor, ClapModel
from pinecone import Pinecone, ServerlessSpec

import math
import numpy as np

def to_json_safe(v):
    # Convert numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)

    # Handle floats
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # Strings: drop empty/NaN-like
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        # turn "True"/"False" strings into booleans
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        return s

    # Booleans/ints OK
    if isinstance(v, (bool, int)):
        return v

    # Lists: keep only JSON-safe children
    if isinstance(v, (list, tuple)):
        cleaned = [to_json_safe(x) for x in v]
        cleaned = [x for x in cleaned if x is not None]
        return cleaned

    # Dicts or anything else: skip to be safe
    return None

def clean_metadata(md: dict) -> dict:
    safe = {}
    for k, v in md.items():
        j = to_json_safe(v)
        if j is not None:
            safe[k] = j
    return safe

# -----------------------------
# Defaults (override via CLI)
# -----------------------------
DEFAULT_INDEX = "music-clap-512"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
AUDIO_SR = 48000
AUDIO_SECONDS = 30
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L2NORM = True

KEEP_META = [
    "id","title","artist","album","path","duration","explicit","gender",
    "danceability","timbre","tagged_genre","rosamerica_top3","moods","popularity"
]

# -----------------------------
# Utilities
# -----------------------------
def l2norm(x: np.ndarray) -> np.ndarray:
    if not L2NORM:
        return x
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom

def safe_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    s = str(x).strip().lower()
    return s in {"true","t","1","yes","y"}

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def parse_listish(x):
    """Parse "['A','B']" or 'A, B, C' -> list[str]; dicts are returned as dict."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(i) for i in v]
        if isinstance(v, dict):
            return {str(k): str(vv) for k, vv in v.items()}
    except Exception:
        pass
    return [p.strip() for p in s.split(",") if p.strip()]

def load_audio_48k_mono(path: str, seconds=AUDIO_SECONDS):
    # Prefer soundfile, fallback to librosa
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sr != AUDIO_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=AUDIO_SR)
    except Exception:
        wav, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
    target_len = AUDIO_SR * seconds
    if len(wav) >= target_len:
        start = (len(wav) - target_len) // 2
        wav = wav[start:start + target_len]
    else:
        pad = target_len - len(wav)
        left, right = pad // 2, pad - (pad // 2)
        wav = np.pad(wav, (left, right), mode="constant")
    return wav.astype(np.float32), AUDIO_SR

def build_text_summary(row: pd.Series) -> str:
    genres = parse_listish(row.get("tagged_genre"))
    top3 = parse_listish(row.get("rosamerica_top3"))
    moods = row.get("moods")
    mood_str = ""
    if isinstance(moods, str):
        try: moods = ast.literal_eval(moods)
        except Exception: pass
    if isinstance(moods, dict):
        pos = [k for k, v in moods.items()
               if str(v).lower().startswith("not ") is False and str(v).lower() not in {"false","0","no"}]
        mood_str = ", ".join(sorted(set(pos))) if pos else ""
    elif isinstance(moods, str):
        mood_str = moods

    bits = [
        f"title: {row.get('title')}",
        f"artist: {row.get('artist')}",
        f"album: {row.get('album')}",
        f"timbre: {row.get('timbre')}",
    ]
    if genres: bits.append("genres: " + ", ".join(genres))
    if isinstance(top3, list) and top3: bits.append("style: " + ", ".join(top3))
    if mood_str: bits.append("moods: " + mood_str)
    if row.get("danceability"): bits.append(f"danceability: {row.get('danceability')}")
    if row.get("gender"): bits.append(f"vocal: {row.get('gender')}")
    return " | ".join(bits)

# -----------------------------
# CLAP init & embed
# -----------------------------
def load_clap():
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
    model.eval()
    return processor, model

@torch.no_grad()
def embed_audio(paths, processor, model):
    feats = []
    for p in paths:
        wav, sr = load_audio_48k_mono(p)
        inputs = processor(audios=wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        emb = model.get_audio_features(**inputs)
        if L2NORM:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        feats.append(emb.cpu().numpy()[0])
    return np.stack(feats, axis=0)

@torch.no_grad()
def embed_text(texts, processor, model):
    feats = []
    for t in texts:
        inputs = processor(text=[t], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        emb = model.get_text_features(**inputs)
        if L2NORM:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        feats.append(emb.cpu().numpy()[0])
    return np.stack(feats, axis=0)

# -----------------------------
# Pinecone
# -----------------------------
def get_index(name, cloud, region, dim=512):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY env var is not set.")
    pc = Pinecone(api_key=api_key)
    if name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc.Index(name)

# -----------------------------
# Main indexing
# -----------------------------
def index_csv(csv_path, index_name, cloud, region, batch=BATCH):
    df = pd.read_csv(csv_path)

    df = df.replace({np.nan: None})

    # Coerce types
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].map(safe_bool)
    if "duration" in df.columns:
        df["duration"] = df["duration"].map(lambda x: safe_float(x, None))
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].map(lambda x: safe_float(x, None))

    # Check files
    missing = [p for p in df["path"].tolist() if not Path(p).exists()]
    if missing:
        print(f"[warn] {len(missing)} files not found (showing up to 5):")
        for m in missing[:5]:
            print("  -", m)

    processor, model = load_clap()
    index = get_index(index_name, cloud, region, dim=512)

    print(f"[info] Indexing {len(df)} rows into Pinecone index '{index_name}' (namespaces: 'audio', 'text')…")
    for start in tqdm(range(0, len(df), batch)):
        chunk = df.iloc[start:start+batch].copy()

        # audio embeddings
        paths = chunk["path"].tolist()
        audio_embs = embed_audio(paths, processor, model)

        # text embeddings from metadata
        texts = [build_text_summary(r) for _, r in chunk.iterrows()]
        text_embs = embed_text(texts, processor, model)

        audio_vectors, text_vectors = [], []

        for i, (_, row) in enumerate(chunk.iterrows()):
            meta = {}
            for k in KEEP_META:
                if k in row:
                    v = row[k]
                    if k in {"tagged_genre","rosamerica_top3"}:
                        v = parse_listish(v)
                    elif k == "moods" and isinstance(v, str):
                        try: v = ast.literal_eval(v)
                        except Exception: pass
                    meta[k] = v

            # normalized fields for filters
            meta["artist_lc"] = str(row.get("artist","")).lower()
            meta["timbre_lc"] = str(row.get("timbre","")).lower()

            meta = clean_metadata(meta)

            rid = str(row["id"])
            audio_vectors.append({
                "id": rid,
                "values": audio_embs[i].astype(np.float32).tolist(),
                "metadata": meta
            })
            text_vectors.append({
                "id": rid,
                "values": text_embs[i].astype(np.float32).tolist(),
                "metadata": meta
            })

        if audio_vectors:
            index.upsert(vectors=audio_vectors, namespace="audio")
        if text_vectors:
            index.upsert(vectors=text_vectors, namespace="text")

    print("[info] Done upserting.")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Embed audio+metadata and upsert to Pinecone (no queries).")
    ap.add_argument("--csv", required=True, help="Path to CSV with columns incl. id,title,artist,album,path,...")
    ap.add_argument("--index", default=DEFAULT_INDEX, help=f"Pinecone index name (default: {DEFAULT_INDEX})")
    ap.add_argument("--cloud", default=DEFAULT_CLOUD, help=f"Pinecone serverless cloud (default: {DEFAULT_CLOUD})")
    ap.add_argument("--region", default=DEFAULT_REGION, help=f"Pinecone serverless region (default: {DEFAULT_REGION})")
    ap.add_argument("--batch", type=int, default=BATCH, help=f"Batch size (default: {BATCH})")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    index_csv(args.csv, args.index, args.cloud, args.region, batch=args.batch)
