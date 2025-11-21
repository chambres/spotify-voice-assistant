import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from difflib import get_close_matches

CANON_GENRES = [
    "Classical", "Dance", "Hip-hop", "Jazz", "Pop",
    "Rhythm and Blues", "Rock", "Speech"
]
GENRE_SYNONYMS = {
    "hiphop": "Hip-hop", "hip-hop": "Hip-hop", "rap": "Hip-hop",
    "r&b": "Rhythm and Blues", "rhythm & blues": "Rhythm and Blues",
    "edm": "Dance", "electronic dance": "Dance",
    "talk": "Speech"
}
MOODS = {"acoustic","aggressive","electronic","happy","party","relaxed","sad"}
GENDER_SYNONYMS = {"male":"male","men":"male","female":"female","women":"female"}

def canon_genre(token: str) -> Optional[str]:
    t = token.strip().lower()
    if t in GENRE_SYNONYMS: return GENRE_SYNONYMS[t]
    # direct hit?
    for g in CANON_GENRES:
        if t == g.lower(): return g
    # fuzzy to your allowed set
    match = get_close_matches(t, [g.lower() for g in CANON_GENRES], n=1, cutoff=0.75)
    return next((g for g in CANON_GENRES if g.lower() == match[0]), None) if match else None

@dataclass
class Popularity:
    op: str
    value: float

@dataclass
class MusicQuery:
    intent: Optional[str] = None                  # "queue" | "play"
    count: Optional[int] = None
    random: bool = False
    like_this: bool = False

    song: Optional[str] = None
    artist: List[str] = field(default_factory=list)
    album: Optional[str] = None

    genders: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    timbre: Optional[str] = None
    tempo: Optional[str] = None                   # "fast" | "slow"
    bpm: Optional[int] = None
    popularity: Optional[Popularity] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "count": self.count,
            "random": self.random,
            "like_this": self.like_this,
            "song": self.song,
            "artist": self.artist,
            "album": self.album,
            "genders": self.genders,
            "genres": self.genres,
            "moods": self.moods,
            "timbre": self.timbre,
            "tempo": self.tempo,
            "bpm": self.bpm,
            "popularity": None if not self.popularity else
                {"op": self.popularity.op, "value": self.popularity.value}
        }

INTENT_RE   = re.compile(r"\b(queue|play)\b", re.I)
COUNT_RE    = re.compile(r"\b(\d+)\s+(?:songs?|tracks?)\b", re.I)
RANDOM_RE   = re.compile(r"\brandom\b", re.I)
LIKETHIS_RE = re.compile(r"\blike\s+(?:this(?:\s+one|\s+song)?)\b", re.I)
SONG_BY_RE  = re.compile(r"(?P<song>.+?)\s+by\s+(?P<artist>[^,]+)$", re.I)

ARTIST_FROM_RE = re.compile(r"\bfrom\s+([a-z0-9 &.'-]+)", re.I)
ALBUM_ON_RE    = re.compile(r"\bon\s+([a-z0-9 &.'-]+)", re.I)
GENDER_RE      = re.compile(r"\b(male|men|female|women)\b", re.I)
BPM_RE         = re.compile(r"\b(\d{2,3})\s*bpm\b", re.I)
TEMPO_RE       = re.compile(r"\b(fast|slow)\b", re.I)
TIMBRE_RE      = re.compile(r"\btimbre\s*(?:is|=)?\s*([a-z-]+)", re.I)
POPULARITY_RE  = re.compile(
    r"\bpopularity\s*(>=|<=|>|<|=|above|below|equal(?:\s*to)?)\s*(1(?:\.0)?|0?\.\d+)\b", re.I
)

# capture explicit genres anywhere; we’ll map to canon set
GENRE_WORDS = sorted(
    set(list(GENRE_SYNONYMS.keys()) + [g.lower() for g in CANON_GENRES] + ["hip hop"]),
    key=len, reverse=True
)
GENRE_ANY_RE = re.compile(r"\b(" + "|".join(map(re.escape, GENRE_WORDS)) + r")\b", re.I)

MOODS_ANY_RE = re.compile(r"\b(" + "|".join(MOODS) + r")\b", re.I)

def normalize(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("hip hop", "hip-hop")
    return t

def parse_popularity(m: re.Match) -> Popularity:
    op_raw, val_raw = m.group(1).lower(), float(m.group(2))
    op = { "above": ">", "below": "<", "equal": "=", "equal to": "=" }.get(op_raw, op_raw)
    return Popularity(op=op, value=val_raw)

def parse_command(text: str) -> MusicQuery:
    q = MusicQuery()
    s = normalize(text)

    # short-circuit: "song by artist"
    m = SONG_BY_RE.search(s)
    if m:
        q.intent = (INTENT_RE.search(s).group(1).lower() if INTENT_RE.search(s) else "queue")
        q.song = m.group("song").strip()
        q.artist = [m.group("artist").strip()]
        q.count = 1
        return q

    # intent & count
    m = INTENT_RE.search(s)
    if m: q.intent = m.group(1).lower()
    m = COUNT_RE.search(s)
    if m: q.count = int(m.group(1))

    # flags
    q.random   = bool(RANDOM_RE.search(s))
    q.like_this = bool(LIKETHIS_RE.search(s))

    # artist / album
    for am in ARTIST_FROM_RE.finditer(s):
        name = am.group(1).strip(" .'-")
        if name and name.lower() not in {"men","women","male","female"}:
            q.artist.append(name)
    m = ALBUM_ON_RE.search(s)
    if m: q.album = m.group(1).strip(" .'-")

    # gender(s)
    for gm in GENDER_RE.finditer(s):
        q.genders.append(GENDER_SYNONYMS[gm.group(1).lower()])

    # genres
    seen = set()
    for gm in GENRE_ANY_RE.finditer(s):
        g = canon_genre(gm.group(1))
        if g and g not in seen:
            q.genres.append(g); seen.add(g)

    # moods
    seen_m = set()
    for mm in MOODS_ANY_RE.finditer(s):
        mword = mm.group(1).lower()
        if mword in MOODS and mword not in seen_m:
            q.moods.append(mword); seen_m.add(mword)

    # timbre, tempo, bpm, popularity
    m = TIMBRE_RE.search(s)
    if m: q.timbre = m.group(1).lower()
    m = TEMPO_RE.search(s)
    if m: q.tempo = m.group(1).lower()
    m = BPM_RE.search(s)
    if m: q.bpm = int(m.group(1))
    m = POPULARITY_RE.search(s)
    if m: q.popularity = parse_popularity(m)

    # defaults
    if q.count is None: q.count = 1
    if "popular" in s and q.popularity is None:
        q.popularity = Popularity(op=">", value=0.7)

    return q

print(parse_command("queue a fast male drake song from views"))