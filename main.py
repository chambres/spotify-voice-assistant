from song_downloader.eng_downloader import download_songs_for_testing

from understanding_engine.main import make_query_with_llm
from vectorization_engine.hybrid_query import hybrid_query_from_spec, pretty_print
from spotify_control.main import control_spotify

import random

# for testing (download and vectorize):
csv_path = download_songs_for_testing("https://open.spotify.com/playlist/3mdTRaYBp5fnNal7H4vrJX")

query = input("Enter your query: ")

parsed_query = make_query_with_llm(query)


rows = None

# `parsed_query` is expected to be a dict matching your JSON schema.
if not isinstance(parsed_query, dict):
	print("[error] parsed_query must be a dict. Got:", type(parsed_query))
	exit(1)


rows = hybrid_query_from_spec(parsed_query)
if not rows:
	print("No matches found or could not resolve a track id from the query spec.")
else:
	# Decide action (play/queue)
	action = str(parsed_query.get("action", "play")).lower()

	# count may be an int or string; default to 1 or len(rows)
	try:
		count = int(parsed_query.get("count", 1) or 1)
	except Exception:
		count = 1

	# random may be boolean or string
	def truthy(v):
		if isinstance(v, bool):
			return v
		if v is None:
			return False
		s = str(v).strip().lower()
		return s in {"true", "1", "yes", "y"}

	do_random = truthy(parsed_query.get("random", False))

	ids = [r.get("id") for r in rows if r.get("id")]
	if not ids:
		print("[error] no track ids in rows returned by hybrid query.")
		exit(1)

	if do_random:
		random.shuffle(ids)

	selected = ids[:max(1, count)]

	# Optional device id in spec
	device_id = parsed_query.get("device_id")

	print(f"Performing Spotify action '{action}' on {len(selected)} track(s):", selected)
	try:
		res = control_spotify(action, selected, device_id=device_id)
		print("Spotify API result:", res)
	except Exception as e:
		print(f"[error] control_spotify failed: {e}")


    









