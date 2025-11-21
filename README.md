# spotifyAI

A small personal project for downloading tracks, extracting embeddings with CLAP, indexing/upserting them into Pinecone, and controlling Spotify playback via the Spotify Web API.

## What this repo contains

- `song_downloader/` — download audio (yt-dlp), fetch metadata, extract local acoustic features.
- `vectorization_engine/` — batch and hybrid query helpers for embedding and searching (CLAP + Pinecone).
- `spotify_control/` — simple OAuth helper + functions to play/queue tracks using your Spotify account.
- `app/` — convenience scripts for auth and local testing.
- `main.py` — example workflow: download a playlist, run LLM parsing, run hybrid query, and control Spotify playback.
- `tracks.csv` — (now tracked) optional CSV of tracks that can be bulk-indexed with `vectorization_engine/upload.py`.

## Quickstart (Windows PowerShell)

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install core dependencies (adjust if you need GPU support):

```powershell
python -m pip install --upgrade pip
# CPU-only (simpler):
python -m pip install python-dotenv transformers requests pinecone-client soundfile librosa yt_dlp tqdm

# OR, for AMD GPU on Windows, install DirectML build of PyTorch (recommended for AMD):
python -m pip install --upgrade pip
python -m pip install torch-directml
# then install the rest
python -m pip install python-dotenv transformers requests pinecone-client soundfile librosa yt_dlp tqdm
```

3. Configure secrets and API keys

- Create `app/.env` with your Spotify app credentials (do not commit):

```
CLIENT_ID=your_spotify_client_id
CLIENT_SECRET=your_spotify_client_secret
REDIRECT_URI=http://127.0.0.1:8080/callback
```

- If you use Pinecone, set `PINECONE_API_KEY` in your environment or in the appropriate `.env` (e.g. top-level or `vectorization_engine/.env`).

4. Register Redirect URI in Spotify Developer Dashboard

- On the Spotify Developer Dashboard, open the app with your `CLIENT_ID` and add the exact redirect URI: `http://127.0.0.1:8080/callback`.

5. Run the example workflow

```powershell
python .\main.py
```

The flow will:
- Download and embed a small playlist (example used in the script).
- Prompt you for a natural language query (parsed by your LLM helper in `understanding_engine`).
- Run the hybrid similarity search and then call the Spotify API to play or queue tracks.

