#!/usr/bin/env python3
"""
Hybrid similarity query: audio vector + metadata-aware re-ranking.

It:
  1) Queries Pinecone by ID in the 'audio' namespace (and optionally 'text')
  2) Computes metadata overlaps (genres, rosamerica_top3, timbre, gender)
  3) Produces a weighted score you can tune via flags

Usage examples:
  # default: audio-heavy, metadata-aware
  python hybrid_query.py --id 5TXQCMKN6TgemTL3c4wRTn --topk 10

  # stronger genre/top3 influence, include text namespace too
  python hybrid_query.py --id 5TXQCMKN6TgemTL3c4wRTn --use-text --w-audio 0.6 --w-text 0.15 --w-genre 0.15 --w-top3 0.1

  # require at least 1 shared genre to keep results
  python hybrid_query.py --id <ID> --min-genre-overlap 0.01
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import os, argparse, ast
from typing import Any, Dict, List, Optional, Tuple
from pinecone import Pinecone

DEFAULT_INDEX = "music-clap-512"
AUDIO_NS = "audio"
TEXT_NS  = "text"

def get_index(name: str):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(name)

def normalize_cosine(s: float) -> float:
    # Pinecone returns cosine sim; map [-1, 1] -> [0, 1]
    return max(0.0, min(1.0, (float(s) + 1.0) / 2.0))

def as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(i).strip() for i in v if str(i).strip()]
    except Exception:
        pass
    return [p.strip() for p in s.split(",") if p.strip()]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(map(str.lower, a)), set(map(str.lower, b))
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def overlap_ratio(a: List[str], b: List[str], denom: int) -> float:
    A, B = set(map(str.lower, a)), set(map(str.lower, b))
    return min(1.0, len(A & B) / float(max(1, denom)))

def fetch_md(index, vec_id: str, namespace: str) -> Dict[str, Any]:
    fr = index.fetch(ids=[vec_id], namespace=namespace)
    # v3 typed response: use .vectors dict
    v = fr.vectors.get(vec_id)
    if not v:
        raise ValueError(f"ID '{vec_id}' not found in namespace '{namespace}'.")
    return (v.metadata or {})

def get_matches(index, vec_id: str, namespace: str, top_k: int) -> Dict[str, Tuple[float, Dict[str, Any]]]:
    """Return mapping id -> (score, metadata) for a single namespace query-by-id."""
    qr = index.query(id=vec_id, top_k=top_k, namespace=namespace, include_metadata=True)
    out: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for m in (qr.matches or []):
        out[m.id] = (float(m.score), m.metadata or {})
    return out

def hybrid_query(
    index_name: str,
    song_id: str,
    top_k: int = 10,
    cand_k: int = 100,
    use_text: bool = True,
    w_audio: float = 0.7,
    w_text: float = 0.1,
    w_genre: float = 0.12,
    w_top3: float = 0.08,
    w_timbre: float = 0.03,
    w_gender: float = 0.02,
    min_genre_overlap: float = 0.0,   # drop if genre Jaccard < this
) -> List[Dict[str, Any]]:
    """
    Build a candidate set then re-rank with a weighted score combining:
      - audio cosine sim (normalized)
      - text cosine sim (normalized, if use_text)
      - genre Jaccard
      - rosamerica_top3 overlap (0..1; 3 = perfect)
      - timbre equality (0/1)
      - gender equality (0/1)
    """
    index = get_index(index_name)

    # base metadata from audio namespace (same as text, but we’ll assume audio exists)
    base_md = fetch_md(index, song_id, AUDIO_NS)
    base_genres = as_list(base_md.get("tagged_genre"))
    base_top3   = as_list(base_md.get("rosamerica_top3"))
    base_timbre = str(base_md.get("timbre_lc") or base_md.get("timbre") or "").lower().strip()
    base_gender = str(base_md.get("gender") or "").strip().lower()

    # fetch candidates from audio (and text optionally)
    audio_map = get_matches(index, song_id, AUDIO_NS, cand_k)
    text_map  = get_matches(index, song_id, TEXT_NS,  cand_k) if use_text else {}

    # unify candidate IDs
    cand_ids = set(audio_map.keys()) | set(text_map.keys())
    if song_id in cand_ids:
        cand_ids.remove(song_id)  # exclude the query track

    ranked = []
    for cid in cand_ids:
        a_score, a_md = audio_map.get(cid, (0.0, {}))
        t_score, t_md = text_map.get(cid, (0.0, {}))
        md = a_md or t_md or {}

        # normalize vector similarities to [0,1]
        a = normalize_cosine(a_score)
        t = normalize_cosine(t_score) if use_text else 0.0

        # metadata overlaps
        genres = as_list(md.get("tagged_genre"))
        top3   = as_list(md.get("rosamerica_top3"))
        timbre = str(md.get("timbre_lc") or md.get("timbre") or "").lower().strip()
        gender = str(md.get("gender") or "").strip().lower()

        g_sim = jaccard(base_genres, genres)             # 0..1
        r_sim = overlap_ratio(base_top3, top3, denom=3)  # 0..1
        timbre_eq = 1.0 if (base_timbre and timbre and base_timbre == timbre) else 0.0
        gender_eq = 1.0 if (base_gender and gender and base_gender == gender) else 0.0

        # optional hard filter by minimal genre overlap
        if g_sim < min_genre_overlap:
            continue

        final = (
            w_audio * a +
            w_text  * t +
            w_genre * g_sim +
            w_top3  * r_sim +
            w_timbre * timbre_eq +
            w_gender * gender_eq
        )

        ranked.append({
            "id": cid,
            "score_audio": a,
            "score_text": t,
            "genre_sim": g_sim,
            "top3_sim": r_sim,
            "timbre_eq": timbre_eq,
            "gender_eq": gender_eq,
            "final_score": final,
            **md
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_k]

def pretty_print(rows: List[Dict[str, Any]]):
    for i, r in enumerate(rows, 1):
        title  = r.get("title", "")
        artist = r.get("artist", "")
        album  = r.get("album", "")
        fs     = r.get("final_score", 0.0)
        comps  = f"a={r['score_audio']:.3f}, t={r['score_text']:.3f}, g={r['genre_sim']:.2f}, r3={r['top3_sim']:.2f}"
        print(f"{i:2d}. {title} — {artist} [{album}]  (score={fs:.4f}, {comps})   id={r['id']}")


def find_track_id_by_metadata(index, title: str, artist: Optional[str] = None, namespace: str = TEXT_NS) -> Optional[str]:
    """Try to find a track id by exact metadata match on title and artist.

    Returns the first matching id or None if not found.
    """
    if not title:
        return None
    # Build a metadata filter for Pinecone (exact-match fields)
    filt = {"title": title}
    if artist:
        filt["artist"] = artist

    # query by filter (no vector) using a small top_k to get candidates
    try:
        qr = index.query(top_k=1, namespace=namespace, filter=filt, include_values=False)
        for m in (qr.matches or []):
            return m.id
    except Exception:
        # Some Pinecone deployments may not support filter-only queries; fall back to fetch-by-metadata is not available.
        return None
    return None


def hybrid_query_from_spec(spec: Dict[str, Any], index_name: str = DEFAULT_INDEX) -> List[Dict[str, Any]]:
    """Wrapper to call `hybrid_query` using a parsed intent/spec dictionary.

    Expected keys (based on your JSON schema):
      - `title` (str)
      - `artists` -> `artists_names` (list[str])
      - `count` (int) -> maps to `top_k`
      - other gating/weights currently defaulted

    The function will try to resolve a track id by matching `title` and first artist
    against the index metadata (text namespace). If not found, returns an empty list.
    """
    idx = get_index(index_name)

    title = spec.get("title") if isinstance(spec.get("title"), str) else None
    artists = []
    ablock = spec.get("artists") or {}
    if isinstance(ablock, dict):
        artists = ablock.get("artists_names") or []
    elif isinstance(spec.get("artists"), list):
        artists = spec.get("artists")
    first_artist = artists[0] if artists else None

    # Attempt to find a matching track id by metadata
    track_id = find_track_id_by_metadata(idx, title=title, artist=first_artist, namespace=TEXT_NS)
    if not track_id:
        # Try fallback: lookup by title only
        track_id = find_track_id_by_metadata(idx, title=title, artist=None, namespace=TEXT_NS)

    if not track_id:
        print(f"[warn] Could not resolve a track id for title='{title}' artist='{first_artist}'")
        return []

    top_k = int(spec.get("count", 10) or 10)

    # You can map additional spec fields to the hybrid_query weights if desired.
    return hybrid_query(
        index_name=index_name,
        song_id=track_id,
        top_k=top_k,
        cand_k=100,
        use_text=True,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid audio+metadata similarity search.")
    ap.add_argument("--id", required=True, help="Track ID to query by")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cands", type=int, default=100, help="Candidate pool size per namespace")
    ap.add_argument("--use-text", action="store_true", help="Also use the 'text' namespace scores")
    # weights
    ap.add_argument("--w-audio", type=float, default=0.7)
    ap.add_argument("--w-text",  type=float, default=0.1)
    ap.add_argument("--w-genre", type=float, default=0.12)
    ap.add_argument("--w-top3",  type=float, default=0.08)
    ap.add_argument("--w-timbre", type=float, default=0.03)
    ap.add_argument("--w-gender", type=float, default=0.02)
    # gating
    ap.add_argument("--min-genre-overlap", type=float, default=0.0,
                    help="Drop candidates whose genre Jaccard is below this (e.g., 0.05)")
    args = ap.parse_args()

    rows = hybrid_query(
        index_name=args.index,
        song_id=args.id,
        top_k=args.topk,
        cand_k=args.cands,
        use_text=args.use_text,
        w_audio=args.w_audio,
        w_text=args.w_text,
        w_genre=args.w_genre,
        w_top3=args.w_top3,
        w_timbre=args.w_timbre,
        w_gender=args.w_gender,
        min_genre_overlap=args.min_genre_overlap,
    )
    pretty_print(rows)
