# filename: spotify_browser.py
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QDateTime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLineEdit, QPlainTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineCookieStore
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtNetwork import QNetworkCookie

# Capture cookies associated with these domains (host-only or subdomains)
SPOTIFY_DOMAINS = {"open.spotify.com", "spotify.com"}
DEFAULT_START_URL = "https://open.spotify.com/"

# ------------- helpers: cookie conversions & filtering ----------------
def cookie_to_dict(cookie: QNetworkCookie) -> dict:
    exp_dt = cookie.expirationDate()
    exp_iso = exp_dt.toUTC().toString(Qt.ISODate) if exp_dt.isValid() else None
    return {
        "name": bytes(cookie.name()).decode("utf-8", errors="replace"),
        "value": bytes(cookie.value()).decode("utf-8", errors="replace"),
        "domain": cookie.domain(),
        "path": cookie.path(),
        "expiration_utc": exp_iso,  # None => session cookie
        "is_secure": cookie.isSecure(),
        "is_http_only": cookie.isHttpOnly(),
        "same_site": None,  # Qt6 doesn't expose SameSite on QNetworkCookie
    }

def dict_to_cookie(d: dict) -> QNetworkCookie:
    c = QNetworkCookie()
    c.setName(d.get("name", "").encode("utf-8"))
    c.setValue(d.get("value", "").encode("utf-8"))
    if d.get("domain"):
        c.setDomain(d["domain"])
    if d.get("path"):
        c.setPath(d["path"])

    # expiration
    exp = d.get("expiration_utc")
    if exp:
        # QDateTime.fromString expects ISO format; handle both with/without 'Z'
        qdt = QDateTime.fromString(exp, Qt.ISODate)
        if qdt.isValid():
            c.setExpirationDate(qdt)
    # flags
    if d.get("is_secure") is True:
        c.setSecure(True)
    if d.get("is_http_only") is True:
        c.setHttpOnly(True)
    return c

def cookie_key(cookie: QNetworkCookie) -> tuple:
    return (bytes(cookie.name()), cookie.domain(), cookie.path())

def is_spotify_cookie(cookie: QNetworkCookie) -> bool:
    dom = (cookie.domain() or "").lstrip(".").lower()
    return any(dom == d or dom.endswith("." + d) for d in SPOTIFY_DOMAINS)

# ----------------------------------------------------------------------

class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spotify Cookie Helper (PySide6 + QtWebEngine)")
        self.resize(1200, 820)

        # --- UI: top bar
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("Enter URL (e.g., https://open.spotify.com/)")
        self.url_edit.returnPressed.connect(self.load_from_urlbar)

        self.btn_back = QPushButton("← Back")
        self.btn_fwd = QPushButton("→ Forward")
        self.btn_print = QPushButton("Print Cookies")
        self.btn_save_all = QPushButton("Save ALL Cookies")
        self.btn_save_pre = QPushButton("Save Pre-Login")
        self.btn_save_post = QPushButton("Save Post-Login (diff)")
        self.btn_load = QPushButton("Load Cookies")

        top = QWidget()
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.setSpacing(8)
        top_l.addWidget(self.btn_back)
        top_l.addWidget(self.btn_fwd)
        top_l.addWidget(self.url_edit, 1)
        top_l.addWidget(self.btn_print)
        top_l.addWidget(self.btn_save_all)
        top_l.addWidget(self.btn_save_pre)
        top_l.addWidget(self.btn_save_post)
        top_l.addWidget(self.btn_load)

        # --- Web view
        self.view = QWebEngineView()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top)
        layout.addWidget(self.view, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # --- Profile & cookie store
        self.profile = QWebEngineProfile.defaultProfile()
        self.cookie_store: QWebEngineCookieStore = self.profile.cookieStore()

        self._spotify_cookies = {}  # key -> QNetworkCookie
        self._pre_login_snapshot = None

        # receive cookie events
        self.cookie_store.cookieAdded.connect(self._on_cookie_added)
        self.cookie_store.cookieRemoved.connect(self._on_cookie_removed)
        self.cookie_store.loadAllCookies()

        # wire buttons
        self.btn_back.clicked.connect(self.view.back)
        self.btn_fwd.clicked.connect(self.view.forward)
        self.btn_print.clicked.connect(self.print_spotify_cookies)
        self.btn_save_all.clicked.connect(self.save_spotify_cookies_all)
        self.btn_save_pre.clicked.connect(self.save_pre_login_snapshot)
        self.btn_save_post.clicked.connect(self.save_post_login_diff)
        self.btn_load.clicked.connect(self.load_cookies_from_file)

        # nav updates (Qt6: no history.changed)
        self.view.urlChanged.connect(self._sync_url_bar)
        self.view.urlChanged.connect(self._update_nav_buttons)
        self.view.loadFinished.connect(lambda _: self._update_nav_buttons())

        # start
        self.url_edit.setText(DEFAULT_START_URL)
        self.view.load(QUrl(DEFAULT_START_URL))
        self._update_nav_buttons()

    # ------------------------- cookie tracking -------------------------
    def _on_cookie_added(self, cookie: QNetworkCookie):
        if is_spotify_cookie(cookie):
            self._spotify_cookies[cookie_key(cookie)] = QNetworkCookie(cookie)

    def _on_cookie_removed(self, cookie: QNetworkCookie):
        if is_spotify_cookie(cookie):
            self._spotify_cookies.pop(cookie_key(cookie), None)

    def _collect_spotify_cookie_dicts(self) -> list[dict]:
        return [cookie_to_dict(c) for c in self._spotify_cookies.values()]

    # ------------------------- nav helpers -----------------------------
    def load_from_urlbar(self):
        raw = self.url_edit.text().strip()
        if not raw:
            return
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
        self.view.load(QUrl(raw))

    def _sync_url_bar(self, qurl: QUrl):
        self.url_edit.blockSignals(True)
        self.url_edit.setText(qurl.toString())
        self.url_edit.blockSignals(False)

    def _update_nav_buttons(self):
        hist = self.view.history()
        self.btn_back.setEnabled(hist.canGoBack())
        self.btn_fwd.setEnabled(hist.canGoForward())

    # ------------------------- UI actions ------------------------------
    def print_spotify_cookies(self):
        cookies = self._collect_spotify_cookie_dicts()
        if not cookies:
            QMessageBox.information(self, "Cookies", "No Spotify cookies yet.")
            return
        text = json.dumps(cookies, indent=2, ensure_ascii=False)
        viewer = QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        viewer.setMinimumSize(800, 600)
        viewer.setWindowTitle("Spotify Cookies (Session Snapshot)")
        viewer.show()

    def save_spotify_cookies_all(self):
        cookies = self._collect_spotify_cookie_dicts()
        if not cookies:
            QMessageBox.information(self, "Save Cookies", "No Spotify cookies to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ALL Spotify Cookies", "spotify_cookies.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cookies, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save cookies:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved {len(cookies)} cookies:\n{path}")

    def save_pre_login_snapshot(self):
        cookies = self._collect_spotify_cookie_dicts()
        self._pre_login_snapshot = {c["name"]: c for c in cookies}
        QMessageBox.information(
            self, "Pre-Login Saved",
            f"Captured {len(cookies)} pre-login Spotify cookies in memory."
        )

    def save_post_login_diff(self):
        if self._pre_login_snapshot is None:
            QMessageBox.warning(self, "Missing Snapshot",
                                "Click 'Save Pre-Login' before logging in.")
            return
        now = {c["name"]: c for c in self._collect_spotify_cookie_dicts()}
        diff = {}
        for name, c in now.items():
            prev = self._pre_login_snapshot.get(name)
            if prev is None or prev.get("value") != c.get("value"):
                diff[name] = c

        if not diff:
            QMessageBox.information(self, "No Changes",
                                    "No new/changed Spotify cookies since pre-login.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Post-Login Cookies (diff)", "spotify_postlogin.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(list(diff.values()), f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save post-login cookies:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved {len(diff)} post-login cookies:\n{path}")

    def load_cookies_from_file(self):
        """
        Load cookies from a JSON file (in the same format we save) and insert
        them into the current WebEngine profile's cookie store.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Cookies JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # support dict of name->cookie or a list of cookie dicts
                cookies_list = list(data.values())
            elif isinstance(data, list):
                cookies_list = data
            else:
                raise ValueError("Unrecognized JSON structure (expect list or dict).")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to read JSON:\n{e}")
            return

        # Insert cookies. Origin URL is used by Chromium to scope security policies.
        # Use the cookie's domain when available; fall back to open.spotify.com.
        inserted = 0
        for cd in cookies_list:
            try:
                c = dict_to_cookie(cd)
                domain = (cd.get("domain") or "open.spotify.com").lstrip(".")
                origin = QUrl(f"https://{domain}/")
                # setCookie(cookie, origin)
                self.cookie_store.setCookie(c, origin)
                inserted += 1
            except Exception:
                # keep going on individual errors
                pass

        QMessageBox.information(self, "Loaded", f"Inserted {inserted} cookies into the profile.")

        # Helpful: navigate (or refresh) to ensure they apply where needed
        if not self.view.url().toString().startswith("https://open.spotify.com"):
            self.view.load(QUrl(DEFAULT_START_URL))

# ----------------------------------------------------------------------

def main():
    # Optional: persist profile across runs (uncomment if you want)
    # prof_dir = Path(".qt_profile").resolve()
    # prof_dir.mkdir(exist_ok=True)
    # QWebEngineProfile.defaultProfile().setPersistentStoragePath(str(prof_dir))

    app = QApplication(sys.argv)
    win = BrowserWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
