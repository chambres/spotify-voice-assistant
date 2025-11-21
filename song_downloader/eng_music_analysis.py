#!/usr/bin/env python3
import argparse, json, sys
from typing import Dict, Any, List, Optional

import subprocess
import json
import os

# AcousticBrainz-style ROSAmerica display labels
ROSAMERICA_MAP = {
    "cla": "Classical",
    "dan": "Dance",
    "hip": "Hiphop",
    "jaz": "Jazz",
    "pop": "Pop",
    "rhy": "Rhythm and Blues",
    "roc": "Rock",
    "spe": "Speech",
}

MOOD_KEYS = [
    "mood_acoustic",
    "mood_aggressive",
    "mood_electronic",
    "mood_happy",
    "mood_party",
    "mood_relaxed",
    "mood_sad",
]

def load_json(path: str="output.json") -> Dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def map_rosamerica(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return ROSAMERICA_MAP.get(code, code)

def extract_rosamerica_top3(highlevel: Dict[str, Any]) -> List[str]:
    scores = highlevel.get("genre_rosamerica", {}).get("all", {}) or {}
    top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [map_rosamerica(code) for code, _ in top3]

def extract_tagged_genre(highlevel: Dict[str, Any]) -> Optional[str]:
    return map_rosamerica(highlevel.get("genre_rosamerica", {}).get("value"))

def format_danceability(val: Optional[str]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    if val.startswith("not_"):
        return "Not " + val.split("not_", 1)[1].replace("_", " ")
    # e.g., "danceable" -> "Danceable"
    return val.capitalize()

def present_mood(name: str, val: Optional[str]) -> Optional[str]:
    """
    name: acoustic/aggressive/electronic/happy/party/relaxed/sad
    val:  "acoustic" or "not_acoustic", etc.
    Output examples: "Not acoustic", "Electronic", "Relaxed", "Sad"
    """
    if not isinstance(val, str):
        return None
    if val.startswith("not_"):
        return "Not " + name
    return name.capitalize()

def extract_moods(highlevel: Dict[str, Any]) -> Dict[str, Optional[str]]:
    out = {}
    for k in MOOD_KEYS:
        base = k.replace("mood_", "")  # e.g., "acoustic"
        out[base] = present_mood(base, highlevel.get(k, {}).get("value"))
    return out

def extract_features(ess: Dict[str, Any]) -> Dict[str, Any]:
    hl = ess.get("highlevel", {}) or {}

    return {
        "rosamerica_top3": extract_rosamerica_top3(hl),
        "tagged_genre": extract_tagged_genre(hl),
        "danceability": format_danceability(hl.get("danceability", {}).get("value")),
        "gender": hl.get("gender", {}).get("value"),  # "male" / "female" (lowercase as in example)
        "timbre": (hl.get("timbre", {}) or {}).get("value"),  # may be absent
        "moods": extract_moods(hl),
    }

def windows_path_to_linux_path(win_path: str) -> str:
    #replace \ with /
    return win_path.replace("\\", "/")

def run_essentia(audio_file, output_file="output.json"):
    """
    Run Essentia extractor via Docker and save output JSON.
    """
    work_dir = os.getcwd()
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir}:/work",
        "ghcr.io/mgoltzsche/essentia",
        "essentia_streaming_extractor_music",
        f"./work/{audio_file}",
        f"./work/{output_file}",
        "/etc/essentia/profile.yaml"
    ]
    
    print("Running Essentia extraction...")
    subprocess.run(docker_cmd, check=True)
    print(f"Extraction complete. Results saved in {output_file}.")


def run_on_device_analysis(filepath: str) -> Dict[str, Any]:

    filepath = windows_path_to_linux_path(filepath)

    print(filepath)

    run_essentia(filepath)  # Example usage; adjust as needed

    ess = load_json()
    result = extract_features(ess)

    # Print as JSON (readable). If you truly need Python single-quote repr, swap json.dump for print(result)
    return result
    