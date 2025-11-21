import sys, json

try:
    import webview
except Exception:
    print("pywebview is required. Install with: pip install pywebview")
    raise

URL = sys.argv[1] if len(sys.argv) > 1 else "https://open.spotify.com/"
if not URL.startswith(("http://", "https://")):
    URL = "https://" + URL

def on_loaded():
    w = webview.windows[0]
    js = r"""
    (async () => {
      const result = {
        ua: navigator.userAgent,
        hasMediaSource: !!window.MediaSource,
        aac_mp4: (window.MediaSource ? MediaSource.isTypeSupported('audio/mp4; codecs="mp4a.40.2"') : false),
        hasEME: !!navigator.requestMediaKeySystemAccess
      };
      return result;
    })();
    """
    try:
        info = w.evaluate_js(js)
        print("[pywebview diagnostics]", json.dumps(info, indent=2))
    except Exception as e:
        print("[pywebview diagnostics] JS eval failed:", e)

# Create the window
webview.create_window("Spotify (System WebView)", URL, easy_drag=False)

# Run with Edge WebView2 backend
webview.start(on_loaded, gui="edgechromium", debug=True)
