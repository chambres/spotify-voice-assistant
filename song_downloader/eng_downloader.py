#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .eng_acousticbrainz import get_acoustic_features, extract_music_summary
from .eng_spotify_metadata import get_track_info, get_songs_in_playlist
from .eng_music_analysis import run_on_device_analysis

from dotenv import load_dotenv
load_dotenv()

import os
import ast
import csv
from pprint import pprint
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

from transformers import ClapProcessor, ClapModel
from pinecone import Pinecone, ServerlessSpec

from yt_dlp import YoutubeDL

# -----------------------------
# Pinecone / CLAP settings
# -----------------------------
DEFAULT_INDEX  = "music-clap-512"
DEFAULT_CLOUD  = "aws"
DEFAULT_REGION = "us-east-1"
AUDIO_SR       = 48000
AUDIO_SECONDS  = 30
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
L2NORM         = True

KEEP_META = [
    "id","title","artist","album","path","duration","explicit","gender",
    "danceability","timbre","tagged_genre","rosamerica_top3","moods","popularity"
]

def to_json_safe(v):
    import math as _math
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if _math.isnan(v) or _math.isinf(v):
            return None
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan","none","null"}:
            return None
        if s.lower() in {"true","false"}:
            return s.lower() == "true"
        return s
    if isinstance(v, (bool, int)):
        return v
    if isinstance(v, (list, tuple)):
        cleaned = [to_json_safe(x) for x in v]
        cleaned = [x for x in cleaned if x is not None]
        return cleaned
    return None

def clean_metadata(md: dict) -> dict:
    safe = {}
    for k, v in md.items():
        j = to_json_safe(v)
        if j is not None:
            safe[k] = j
    return safe

def parse_listish(x):
    """Parse "['A','B']" or 'A, B, C' -> list[str]; dicts returned as dict."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
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

def build_text_summary(row: dict) -> str:
    genres = parse_listish(row.get("tagged_genre"))
    top3   = parse_listish(row.get("rosamerica_top3"))
    moods  = row.get("moods")
    mood_str = ""
    if isinstance(moods, str):
        try:
            moods = ast.literal_eval(moods)
        except Exception:
            pass
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

def load_clap():
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
    model.eval()
    return processor, model

@torch.no_grad()
def embed_audio(path, processor, model):
    wav, sr = load_audio_48k_mono(path)
    inputs = processor(audios=wav, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    emb = model.get_audio_features(**inputs)
    if L2NORM:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy()[0].astype(np.float32)

@torch.no_grad()
def embed_text(text, processor, model):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    emb = model.get_text_features(**inputs)
    if L2NORM:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy()[0].astype(np.float32)

def get_index(name=DEFAULT_INDEX, cloud=DEFAULT_CLOUD, region=DEFAULT_REGION, dim=512):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY env var is not set.")
    pc = Pinecone(api_key=api_key)
    existing = [i["name"] for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc.Index(name)

def upsert_row_to_pinecone(row: dict, index, processor, model):
    """Embed audio + text for this row and upsert to Pinecone."""
    # Ensure full audio path exists
    audio_path = row.get("path")
    if not audio_path or not Path(audio_path).exists():
        print(f"[warn] Audio missing for {row.get('id')} -> {audio_path}")
        return

    # Build metadata
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

    # normalized fields for filtering
    meta["artist_lc"] = str(row.get("artist","")).lower()
    meta["timbre_lc"] = str(row.get("timbre","")).lower()
    meta = clean_metadata(meta)

    # Embeddings
    audio_emb = embed_audio(audio_path, processor, model)
    text_emb  = embed_text(build_text_summary(row), processor, model)

    rid = str(row["id"])
    audio_vec = {"id": rid, "values": audio_emb.tolist(), "metadata": meta}
    text_vec  = {"id": rid, "values": text_emb.tolist(),  "metadata": meta}

    index.upsert(vectors=[audio_vec], namespace="audio")
    index.upsert(vectors=[text_vec],  namespace="text")
    print(f"[info] Upserted {rid} to Pinecone (audio + text).")

# -----------------------------
# Your original workflow
# -----------------------------

def download_songs_for_testing(spotify_playlist_url: str):

    # Playlist (you can change this link)
    songs = get_songs_in_playlist(spotify_playlist_url)

    # --- Output CSV path (still optional if you want it as a log) ---
    CSV_PATH = "tracks.csv"
    fieldnames = [
        "id", "title", "artist", "album", "path",
        "duration", "explicit", "gender", "danceability", "timbre",
        "tagged_genre", "rosamerica_top3", "moods", "popularity"
    ]
    f = open(CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    out_dir = "songs"
    os.makedirs(out_dir, exist_ok=True)

    # Init CLAP + Pinecone once
    print("[info] Loading CLAP and connecting to Pinecone…")
    processor, model = load_clap()
    index = get_index(name=DEFAULT_INDEX, cloud=DEFAULT_CLOUD, region=DEFAULT_REGION, dim=512)

    # Use a different playlist for indexing if you want:
    songs = get_songs_in_playlist("https://open.spotify.com/playlist/3mdTRaYBp5fnNal7H4vrJX?si=0eab75abbad34a67")

    for track in songs:


        #if the track is already in pinecone, skip it
        existing = index.fetch(ids=[track['id']], namespace="audio").vectors
        if existing:
            print(f"[info] Skipping {track['id']} as it already exists in Pinecone.")
            continue


        # Spotify metadata
        print("Getting Spotify metadata...")
        track_metadata = get_track_info(track)

        # AcousticBrainz features
        print("Getting AcousticBrainz features...")
        artist = track_metadata.get("artist")
        title  = track_metadata.get("title")
        acoustic_features = get_acoustic_features(artist, title)

        spotify_id = track_metadata["id"]

        if not acoustic_features:
            print("No AcousticBrainz data found!")
        else:
            print("AcousticBrainz data found.")

        # Download via yt-dlp ========================================================
        print("Downloading audio via yt-dlp...")
        wav_stem  = f"{spotify_id}"
        audio_path = os.path.join(out_dir, f"{wav_stem}.wav")

        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            # NOTE: outtmpl WITHOUT extension; FFmpegExtractAudio adds ".wav"
            "outtmpl": os.path.join(out_dir, wav_stem),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "prefer_ffmpeg": True,
            "quiet": True,
            "progress": True,
        }

        query = f"{artist} - {title}"
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=True)
            entry = info["entries"][0] if "entries" in info else info
            print("Saved:", audio_path)

        #==============================================================================

        if not acoustic_features:
            print("Running on-device analysis via Essentia…")
            acoustic_features = run_on_device_analysis(audio_path)

        # Merge
        print("Merging metadata...")
        metadata = {**track_metadata, **(acoustic_features or {})}

        # Build flat row (IMPORTANT: use the FULL audio path here)
        row = {
            "id": spotify_id,
            "title": track_metadata.get("title", ""),
            "artist": track_metadata.get("artist", ""),
            "album": track_metadata.get("album", ""),
            "path": audio_path,  # full path so we can embed it
            "duration": track_metadata.get("duration"),
            "explicit": track_metadata.get("explicit"),
            "gender": (acoustic_features or {}).get("gender"),
            "danceability": (acoustic_features or {}).get("danceability"),
            "timbre": (acoustic_features or {}).get("timbre"),
            "tagged_genre": (acoustic_features or {}).get("tagged_genre"),
            "rosamerica_top3": (acoustic_features or {}).get("rosamerica_top3"),
            "moods": (acoustic_features or {}).get("moods"),
            "popularity": track_metadata.get("popularity"),
        }

        # ---- NEW: Upload to Pinecone before deleting the WAV ----
        try:
            upsert_row_to_pinecone(row, index=index, processor=processor, model=model)
        except Exception as e:
            print(f"[error] Failed to upsert {spotify_id}: {e}")

        # Optional: still log to CSV for auditing
        print("Writing to CSV...")
        writer.writerow(row)
        pprint(row)

        print("Deleting WAV file...")
        try:
            os.remove(audio_path)
            print("Deleted.")
        except Exception as e:
            print(f"[warn] Could not delete file {audio_path}: {e}")

    f.close()
    print(f"✅ CSV saved to {CSV_PATH}")
    print("✅ All tracks processed and uploaded to Pinecone.")

    return CSV_PATH
