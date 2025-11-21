# filename: spotify_browser.py
import json
import sys
import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QDateTime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLineEdit, QPlainTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineCookieStore
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtNetwork import QNetworkCookie
import os, tempfile
from PySide6.QtCore import QUrl
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWebView import QtWebView


# NEW: system (DRM) webview via QtWebView (Edge WebView2/WKWebView)
# NEW: system (DRM) webview via QtWebView (Edge WebView2/WKWebView)
try:
    from PySide6.QtWebView import QtWebView
    from PySide6.QtQuickWidgets import QQuickWidget
    from PySide6.QtCore import QUrl
    _QTWEBVIEW_AVAILABLE = True
except Exception:
    QtWebView = None
    QQuickWidget = None
    _QTWEBVIEW_AVAILABLE = False


# Optional: reduce autoplay friction in QtWebEngine (not a DRM fix)
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--autoplay-policy=no-user-gesture-required")

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
        "same_site": None,
    }

def dict_to_cookie(d: dict) -> QNetworkCookie:
    c = QNetworkCookie()
    c.setName(d.get("name", "").encode("utf-8"))
    c.setValue(d.get("value", "").encode("utf-8"))
    if d.get("domain"):
        c.setDomain(d["domain"])
    if d.get("path"):
        c.setPath(d["path"])
    exp = d.get("expiration_utc")
    if exp:
        qdt = QDateTime.fromString(exp, Qt.ISODate)
        if qdt.isValid():
            c.setExpirationDate(qdt)
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

# ---------------------- DRM-capable WebView window --------------------
class SystemWebViewWindow(QWidget):
    """
    A window that embeds the QtWebView (system WebView) via QML WebView.
    This typically supports AAC + Widevine, so Spotify works.
    """
    def __init__(self, url: str):
        super().__init__()
        self.setWindowTitle("Spotify (System WebView)")
        self.resize(1200, 800)

        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Host QML content inside a QQuickWidget
        self.quick = QQuickWidget(self)
        self.quick.setResizeMode(QQuickWidget.SizeRootObjectToView)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick)

        # Prepare inline QML using QtWebView's WebView
        qml = f"""
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtWebView 1.15

Item {{
    width: 1200
    height: 800
    WebView {{
        id: wv
        anchors.fill: parent
        url: "{url}"
    }}
}}
"""

        # Write QML to a temp file and load it via setSource(...)
        # (PySide6's QQuickWidget.setContent requires QQmlComponent; using a file is simpler.)
        tmp_dir = tempfile.gettempdir()
        self._qml_path = os.path.join(tmp_dir, f"spotify_system_webview_{os.getpid()}.qml")
        with open(self._qml_path, "w", encoding="utf-8") as f:
            f.write(qml)

        self.quick.setSource(QUrl.fromLocalFile(self._qml_path))

        # Optional: simple error check
        def _on_status_changed(status):
            # 1 = Ready, 2 = Loading, 3 = Error
            if status == QQuickWidget.Status.Error:
                QMessageBox.critical(
                    self, "QtWebView Error",
                    "Failed to load the system WebView. "
                    "Ensure 'PySide6-Addons' is installed and WebView2 Runtime is present on Windows."
                )
        self.quick.statusChanged.connect(_on_status_changed)



# ----------------------------------------------------------------------

class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spotify Cookie Helper (QtWebEngine + QtWebView)")
        self.resize(1280, 860)

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

        self.btn_open_system = QPushButton("Open in System WebView (DRM)")
        if not _QTWEBVIEW_AVAILABLE:
            self.btn_open_system.setEnabled(False)
            self.btn_open_system.setToolTip(
                "QtWebView module not available. Install: pip install PySide6-Addons"
            )

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
        top_l.addWidget(self.btn_open_system)

        # --- Web view (QtWebEngine) for cookie work
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
        self.btn_open_system.clicked.connect(self.open_in_system_webview)

        # nav updates (Qt6: no history.changed)
        self.view.urlChanged.connect(self._sync_url_bar)
        self.view.urlChanged.connect(self._update_nav_buttons)
        self.view.loadFinished.connect(lambda _: self._update_nav_buttons())

        # start
        self.url_edit.setText(DEFAULT_START_URL)
        self.view.load(QUrl(DEFAULT_START_URL))
        self._update_nav_buttons()

        # hold refs so extra windows don't get GC'd
        self._system_windows: list[SystemWebViewWindow] = []

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Cookies JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cookies_list = list(data.values())
            elif isinstance(data, list):
                cookies_list = data
            else:
                raise ValueError("Unrecognized JSON structure (expect list or dict).")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to read JSON:\n{e}")
            return

        inserted = 0
        for cd in cookies_list:
            try:
                c = dict_to_cookie(cd)
                domain = (cd.get("domain") or "open.spotify.com").lstrip(".")
                origin = QUrl(f"https://{domain}/")
                self.cookie_store.setCookie(c, origin)
                inserted += 1
            except Exception:
                pass

        QMessageBox.information(self, "Loaded", f"Inserted {inserted} cookies into the profile.")

        if not self.view.url().toString().startswith("https://open.spotify.com"):
            self.view.load(QUrl(DEFAULT_START_URL))

    # --------------------- NEW: system WebView (QtWebView) --------------
    def open_in_system_webview(self):
        if not _QTWEBVIEW_AVAILABLE:
            QMessageBox.warning(
                self, "QtWebView unavailable",
                "Install PySide6-Addons to enable QtWebView (pip install PySide6-Addons)."
            )
            return

        url = self.url_edit.text().strip() or DEFAULT_START_URL
        w = SystemWebViewWindow(url)
        w.show()
        # keep a reference so it isn't garbage-collected
        self._system_windows.append(w)

# ----------------------------------------------------------------------

def main():
    if QtWebView is not None:
        QtWebView.initialize()
    app = QApplication(sys.argv)
    win = BrowserWindow()
    win.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
