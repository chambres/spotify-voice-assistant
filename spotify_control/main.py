import http.server
import socketserver
import threading
import webbrowser
import urllib.parse
import requests
import base64
import json
import os
import time
from typing import List
from dotenv import load_dotenv

# ========= CONFIG (EDIT THESE) =========
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://127.0.0.1:8080/callback")
SCOPES = "user-modify-playback-state"  # add more if you want
TOKENS_FILE = ".spotify_tokens.json"
# ======================================

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1/me/player"

_auth_code_container = {"code": None}


class SpotifyAuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        query_params = urllib.parse.parse_qs(parsed.query)

        if "error" in query_params:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Error returned from Spotify auth.")
            return

        if "code" not in query_params:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No code parameter found in callback.")
            return

        auth_code = query_params["code"][0]
        _auth_code_container["code"] = auth_code

        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Spotify login successful!</h2>"
            b"<p>You can close this window and go back to your terminal.</p>"
            b"</body></html>"
        )


def _start_local_server():
    with socketserver.TCPServer(("localhost", 8080), SpotifyAuthHandler) as httpd:
        httpd.handle_request()  # handle a single request then exit


def _get_authorization_code() -> str:
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "show_dialog": "true",
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    # Debugging: print the exact authorization URL and client/redirect being used
    print("=== Spotify Authorization Request ===")
    print("CLIENT_ID:", CLIENT_ID)
    print("REDIRECT_URI:", REDIRECT_URI)
    print("Authorize URL:", url)
    print("If the browser shows an error, copy/paste the above URL into your browser to inspect the Spotify error page.")

    server_thread = threading.Thread(target=_start_local_server, daemon=True)
    server_thread.start()

    print("Opening browser for Spotify login...")
    webbrowser.open(url, new=1)

    print("Waiting for authorization code...")
    while _auth_code_container["code"] is None:
        time.sleep(0.1)

    return _auth_code_container["code"]


def _exchange_code_for_tokens(auth_code: str) -> dict:
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
    }

    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = requests.post(TOKEN_URL, data=data, headers=headers)
    resp.raise_for_status()
    tokens = resp.json()
    tokens["expires_at"] = time.time() + tokens["expires_in"]
    return tokens


def _refresh_access_token(refresh_token: str) -> dict:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = requests.post(TOKEN_URL, data=data, headers=headers)
    resp.raise_for_status()
    tokens = resp.json()

    # keep the same refresh_token if Spotify doesn't send a new one
    tokens["refresh_token"] = tokens.get("refresh_token", refresh_token)
    tokens["expires_at"] = time.time() + tokens["expires_in"]
    return tokens


def _load_tokens() -> dict | None:
    if not os.path.exists(TOKENS_FILE):
        return None
    with open(TOKENS_FILE, "r") as f:
        return json.load(f)


def _save_tokens(tokens: dict):
    with open(TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def get_access_token() -> str:
    """
    For personal use:
    - First run: opens browser, you log in, saves tokens to .spotify_tokens.json
    - Later runs: silently refreshes access token when needed
    """
    # Fail fast when credentials are missing or left as placeholders.
    if not CLIENT_ID or not CLIENT_SECRET or "YOUR_CLIENT_ID" in CLIENT_ID or "YOUR_CLIENT_SECRET" in CLIENT_SECRET:
        raise RuntimeError("Set CLIENT_ID and CLIENT_SECRET in the script first.")

    tokens = _load_tokens()

    if tokens is None:
        # First-time auth
        code = _get_authorization_code()
        tokens = _exchange_code_for_tokens(code)
        _save_tokens(tokens)
        print("Saved tokens to", TOKENS_FILE)
    else:
        # Check expiry and refresh if needed
        if time.time() > tokens.get("expires_at", 0) - 60:
            print("Access token expired or near expiry, refreshing...")
            tokens = _refresh_access_token(tokens["refresh_token"])
            _save_tokens(tokens)

    return tokens["access_token"]


def control_spotify(action: str, track_ids: List[str], device_id: str | None = None):
    """
    Perform 'play' or 'queue' action on a list of Spotify track IDs (no 'spotify:track:' prefix).

    action: "play" or "queue"
    track_ids: e.g. ["4uLU6hMCjMI75M1A2tKUQC", ...]
    device_id: optional Spotify device id
    """
    access_token = get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    uris = [f"spotify:track:{tid}" for tid in track_ids]

    if action.lower() == "play":
        payload = {"uris": uris}
        device_param = f"?device_id={device_id}" if device_id else ""
        res = requests.put(f"{SPOTIFY_API}/play{device_param}", json=payload, headers=headers)
        print("PLAY:", res.status_code, res.text)
        return res.status_code, res.text

    elif action.lower() == "queue":
        results = []
        for uri in uris:
            params = {"uri": uri}
            if device_id:
                params["device_id"] = device_id
            res = requests.post(f"{SPOTIFY_API}/queue", params=params, headers=headers)
            print("QUEUE:", uri, res.status_code)
            results.append((uri, res.status_code, res.text))
        return results

    else:
        raise ValueError("action must be 'play' or 'queue'")


# Optional: simple CLI usage
if __name__ == "__main__":
    control_spotify("queue", ["4qkVALwOxCIEZ7I5gkZ3m4"])