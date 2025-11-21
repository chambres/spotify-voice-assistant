# pip install websocket-client
import json
import time
import urllib.request
from websocket import create_connection

PORT = 9222  # must match QTWEBENGINE_REMOTE_DEBUGGING used by your app

def get_page_ws_url():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
        targets = json.loads(r.read().decode())
    # Pick a 'page' with your URL/title; adjust the predicate as you wish
    for t in targets:
        if t.get("type") == "page" and ("index.html" in t.get("url", "") or "Pytron" in t.get("title","")):
            return t["webSocketDebuggerUrl"]
    raise RuntimeError("Couldn't find the embedded page target. Is the app running with QTWEBENGINE_REMOTE_DEBUGGING?")

class CDP:
    def __init__(self, ws_url):
        self.ws = create_connection(ws_url)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        self.ws.send(json.dumps(msg))
        # read until we get the matching id
        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == self._id:
                if "error" in resp:
                    raise RuntimeError(f"CDP error for {method}: {resp['error']}")
                return resp.get("result")
            # otherwise it was an event; ignore or handle if you want

def main():
    ws_url = get_page_ws_url()
    cdp = CDP(ws_url)

    # Enable the Runtime domain so we can evaluate JS
    cdp.call("Runtime.enable")

    # Fill the form and click submit using plain JS
    js = r"""
      (function(){
        const byId = id => document.getElementById(id);
        byId('name').value  = 'Grace Hopper';
        byId('email').value = 'grace@example.com';
        byId('submit').click();
        return document.getElementById('out').textContent;
      })();
    """
    result = cdp.call("Runtime.evaluate", {"expression": js, "returnByValue": True})
    print("Automation result:", result["result"]["value"])

    # Example: read back the DOM text
    result2 = cdp.call("Runtime.evaluate", {
        "expression": "document.querySelector('#out').textContent",
        "returnByValue": True
    })
    print("OUT:", result2["result"]["value"])

if __name__ == "__main__":
    main()
