import os, requests
import musicbrainzngs

# --- 1) Configure MusicBrainz (set a proper User-Agent per their guidelines) ---
musicbrainzngs.set_useragent(
    "YourAppName", "1.0", "https://yourapp.example.com/contact"
)

def get_mbid_from_isrc(isrc: str) -> str | None:
    try:
        res = musicbrainzngs.get_recordings_by_isrc(isrc, includes=["artists", "releases"])
        recs = res.get("isrc", {}).get("recording-list", [])
        return recs[0]["id"] if recs else None
    except musicbrainzngs.WebServiceError:
        return None

def search_mbid(artist: str, title: str, duration_ms: int | None = None) -> str | None:
    # MusicBrainz duration is in seconds
    q = {"artist": artist, "recording": title}
    if duration_ms:
        q["dur"] = int(round(duration_ms/1000))
    try:
        res = musicbrainzngs.search_recordings(limit=5, **q)
        recs = res.get("recording-list", [])
        return recs[0]["id"] if recs else None
    except musicbrainzngs.WebServiceError:
        return None

# --- 2) AcousticBrainz helpers ---
AB_BASE = "https://acousticbrainz.org/api/v1"

def ab_low_level(mbid: str, n: int | None = None):
    params = {"n": n} if n is not None else {}
    r = requests.get(f"{AB_BASE}/{mbid}/low-level", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def ab_high_level(mbid: str, n: int | None = None, map_classes: bool = True):
    params = {"n": n, "map_classes": str(map_classes).lower()} if n is not None else {"map_classes": str(map_classes).lower()}
    r = requests.get(f"{AB_BASE}/{mbid}/high-level", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_analysis_from_acousticbrainz(artist: str, title: str, isrc: str | None = None, duration_ms: int | None = None):
    mbid = get_mbid_from_isrc(isrc) if isrc else None
    if not mbid:
        mbid = search_mbid(artist, title, duration_ms)
    if not mbid:
        raise RuntimeError("Could not resolve MBID from MusicBrainz.")

    low = ab_low_level(mbid)      # tempo/key/spectral/rhythm/tonal etc.
    high = ab_high_level(mbid)    # mood/genre/danceability-like tags, etc.
    return {"mbid": mbid, "low_level": low, "high_level": high}

def extract_music_summary(data: dict):
    hl = data.get("highlevel", {})
    md = data.get("metadata", {})
    tags = md.get("tags", {})

    def get_value(feature_name):
        return hl.get(feature_name, {}).get("value")

    # 1) Top 3 genres from the Rosamerica model
    rosa = hl.get("genre_rosamerica", {})
    rosa_all = rosa.get("all", {})
    top3_rosamerica = sorted(rosa_all.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top3_rosamerica = [k for k, _ in top3_rosamerica]

    # 2) Tagged genre from metadata
    tagged_genre_list = tags.get("genre", [])
    tagged_genre = tagged_genre_list[0] if tagged_genre_list else None

    # 3) General features
    danceability = get_value("danceability")
    gender       = get_value("gender")
    timbre       = get_value("timbre")

    # 4) All mood values
    moods = {
        "acoustic": get_value("mood_acoustic"),
        "aggressive": get_value("mood_aggressive"),
        "electronic": get_value("mood_electronic"),
        "happy": get_value("mood_happy"),
        "party": get_value("mood_party"),
        "relaxed": get_value("mood_relaxed"),
        "sad": get_value("mood_sad"),
    }

    return {
        "rosamerica_top3": top3_rosamerica,
        "tagged_genre": tagged_genre,
        "danceability": danceability,
        "gender": gender,
        "timbre": timbre,
        "moods": moods,
    }


def get_acoustic_features(artist: str, title: str):
    try:
        data = get_analysis_from_acousticbrainz(artist, title)
        return extract_music_summary(data["high_level"])
    except Exception as e:
        print(f"There was was an error getting AcousticBrainz data for '{artist} - {title}'")
        return {} # Return empty dict on error


