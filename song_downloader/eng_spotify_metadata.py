import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pprint
from dotenv import load_dotenv
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))




def get_track_info(track_id: str):
    track = sp.track(track_id)
    #get spotify id, artist, album, title, duration
    spotify_id = track["id"]
    artist = track["artists"][0]["name"]
    album = track["album"]["name"]
    title = track["name"]
    duration = track["duration_ms"]
    popularity = track["popularity"]
    explicit = track["explicit"]
    #return as json
    return  {
        "id": spotify_id,
        "artist": artist,
        "album": album,
        "title": title,
        "duration": duration / 1000 , #seconds
        "popularity": popularity / 100, #from 0 to 1
        "explicit": explicit
    }

def get_track_info(track: dict):
    #get spotify id, artist, album, title, duration
    spotify_id = track["id"]
    artist = track["artists"][0]["name"]
    album = track["album"]["name"]
    title = track["name"]
    duration = track["duration_ms"]
    popularity = track["popularity"]
    explicit = track["explicit"]
    #return as json
    return {
        "id": spotify_id,
        "artist": artist,
        "album": album,
        "title": title,
        "duration": duration / 1000 , #seconds
        "popularity": popularity / 100, #from 0 to 1
        "explicit": explicit
    }


def get_songs_in_playlist(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    songs = []
    for item in results["items"]:
        track = item["track"]
        songs.append(track)
    
    return songs
    


#print(get_track_info("https://open.spotify.com/track/5TXQCMKN6TgemTL3c4wRTn"))
get_songs_in_playlist("https://open.spotify.com/playlist/7MPOrL8KzRd4ZYp3pKZ7iH")