#!/usr/bin/env python3
"""
Find songs similar to a given track ID, with optional filters.

Examples:
  # top-10 similar, same artist only
  python query.py --id 5TXQCMKN6TgemTL3c4wRTn --same-artist

  # similar but exclude the same artist and explicit tracks
  python query.py --id 5TXQCMKN6TgemTL3c4wRTn --not-same-artist --no-explicit

  # constrain by album and min popularity
  python query.py --id ... --album "Views" --min-popularity 60
"""

import os
import argparse
from typing import Any, Dict, List, Optional
from pinecone import Pinecone

DEFAULT_INDEX = "music-clap-512"
DEFAULT_NAMESPACE = "audio"

def get_index(name: str):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(name)

def _get(obj, attr: str, default=None):
    """Attribute or dict-style getter (works with v3 typed responses)."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default

def fetch_metadata_by_id(index, vec_id: str, namespace: str) -> Dict[str, Any]:
    """Fetch only metadata for an ID."""
    out = index.fetch(ids=[vec_id], namespace=namespace)
    vectors = _get(out, "vectors", {}) or {}
    v = vectors.get(vec_id)
    if not v:
        raise ValueError(f"ID '{vec_id}' not found in namespace '{namespace}'.")
    md = _get(v, "metadata", {}) or {}
    return md

def build_filter(
    base_md: Optional[Dict[str, Any]] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    same_artist: bool = False,
    not_same_artist: bool = False,
    same_album: bool = False,
    no_explicit: bool = False,
    explicit_only: bool = False,
    min_popularity: Optional[float] = None,
    max_popularity: Optional[float] = None,
    min_danceability: Optional[float] = None,
    max_danceability: Optional[float] = None,
    timbre: Optional[str] = None,
    gender: Optional[str] = None,
    genres: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a Pinecone metadata filter dict.
    Uses normalized fields 'artist_lc' and 'timbre_lc' when applicable.
    """
    f: Dict[str, Any] = {}

    # artist constraints
    if same_artist and base_md:
        a = (base_md.get("artist_lc") or str(base_md.get("artist","")).lower()).strip()
        if a:
            f["artist_lc"] = {"$eq": a}
    if not_same_artist and base_md:
        a = (base_md.get("artist_lc") or str(base_md.get("artist","")).lower()).strip()
        if a:
            f["artist_lc"] = {"$ne": a}
    if artist:
        f["artist_lc"] = {"$eq": artist.lower().strip()}

    # album constraints
    if same_album and base_md:
        alb = base_md.get("album")
        if isinstance(alb, str) and alb.strip():
            f["album"] = {"$eq": alb}
    if album:
        f["album"] = {"$eq": album}

    # explicit
    if no_explicit and not explicit_only:
        f["explicit"] = {"$eq": False}
    if explicit_only and not no_explicit:
        f["explicit"] = {"$eq": True}

    # numeric ranges
    def add_range(field: str, lo: Optional[float], hi: Optional[float]):
        if lo is None and hi is None:
            return
        cond = {}
        if lo is not None:
            cond["$gte"] = float(lo)
        if hi is not None:
            cond["$lte"] = float(hi)
        f[field] = cond

    add_range("popularity", min_popularity, max_popularity)
    add_range("danceability", min_danceability, max_danceability)

    # timbre / gender
    if timbre:
        f["timbre_lc"] = {"$eq": timbre.lower().strip()}
    if gender:
        f["gender"] = {"$eq": gender}

    # genres list (match if any)
    if genres:
        # your metadata stores 'tagged_genre' as a list; $in works as overlap
        f["tagged_genre"] = {"$in": genres}

    return f

def query_similar_by_id(
    index_name: str,
    song_id: str,
    namespace: str = DEFAULT_NAMESPACE,
    top_k: int = 10,
    include_self: bool = False,
    filter_: Optional[Dict[str, Any]] = None,
    metadata_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone using the stored vector for `song_id` and return similar songs.
    """
    index = get_index(index_name)
    k = top_k + (0 if include_self else 1)

    res = index.query(
        id=song_id,
        top_k=k,
        namespace=namespace,
        include_metadata=True,
        filter=filter_,
    )

    matches = _get(res, "matches", []) or []
    if not include_self:
        matches = [m for m in matches if _get(m, "id") != song_id]
    matches = matches[:top_k]

    wanted = set(metadata_fields or [])
    out = []
    for m in matches:
        md = _get(m, "metadata", {}) or {}
        if wanted:
            md = {k: v for k, v in md.items() if k in wanted}
        out.append({
            "id": _get(m, "id"),
            "score": float(_get(m, "score", 0.0)),
            **md,
        })
    return out

def pretty_print(rows: List[Dict[str, Any]]):
    for i, r in enumerate(rows, 1):
        title = r.get("title", "")
        artist = r.get("artist", "")
        album = r.get("album", "")
        score = r.get("score", 0.0)
        rid = r.get("id", "")
        print(f"{i:2d}. {title} — {artist} [{album}] (id={rid}, score={score:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find similar songs by audio embedding, with filters.")
    ap.add_argument("--id", required=True, help="Track ID to query by")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--include-self", action="store_true")

    # Filter options
    ap.add_argument("--same-artist", action="store_true")
    ap.add_argument("--not-same-artist", action="store_true")
    ap.add_argument("--same-album", action="store_true")
    ap.add_argument("--artist", type=str)
    ap.add_argument("--album", type=str)
    ap.add_argument("--no-explicit", action="store_true")
    ap.add_argument("--explicit-only", action="store_true")
    ap.add_argument("--min-popularity", type=float)
    ap.add_argument("--max-popularity", type=float)
    ap.add_argument("--min-danceability", type=float)
    ap.add_argument("--max-danceability", type=float)
    ap.add_argument("--timbre", type=str)
    ap.add_argument("--gender", type=str, choices=["male","female","mixed","instrumental"], help="matches your metadata")
    ap.add_argument("--genre", action="append", dest="genres", help="repeat for multiple genres")

    args = ap.parse_args()

    index = get_index(args.index)

    # When relative filters are requested, fetch the query track's metadata
    base_md = None
    if args.same_artist or args.not_same_artist or args.same_album:
        base_md = fetch_metadata_by_id(index, args.id, args.namespace)

    filter_dict = build_filter(
        base_md=base_md,
        artist=args.artist,
        album=args.album,
        same_artist=args.same_artist,
        not_same_artist=args.not_same_artist,
        same_album=args.same_album,
        no_explicit=args.no_explicit,
        explicit_only=args.explicit_only,
        min_popularity=args.min_popularity,
        max_popularity=args.max_popularity,
        min_danceability=args.min_danceability,
        max_danceability=args.max_danceability,
        timbre=args.timbre,
        gender=args.gender,
        genres=args.genres,
    )

    rows = query_similar_by_id(
        index_name=args.index,
        song_id=args.id,
        namespace=args.namespace,
        top_k=args.topk,
        include_self=args.include_self,
        filter_=filter_dict or None,
        metadata_fields=["title", "artist", "album"],
    )
    pretty_print(rows)
