"""Serve the Connect Four GUI and its compiled neural weights locally."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
import webbrowser

from .structured import compile_structured


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "connect4.flow"
WEB = Path(__file__).with_name("web")


@lru_cache(maxsize=1)
def compiled_game():
    return compile_structured(SOURCE.read_text(), max_steps=1)


@lru_cache(maxsize=1)
def model_bytes():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "connect4.json"
        compiled_game().save(path)
        return path.read_bytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        files = {"/": ("index.html", "text/html"), "/app.mjs": ("app.mjs", "text/javascript"),
                 "/style.css": ("style.css", "text/css"),
                 "/neural.mjs": ("neural.mjs", "text/javascript"),
                 "/game-worker.mjs": ("game-worker.mjs", "text/javascript")}
        if path == "/api/network":
            body, content_type = model_bytes(), "application/json"
        elif path == "/source":
            body, content_type = SOURCE.read_bytes(), "text/plain"
        elif path == "/health":
            body, content_type = json.dumps({"ok": True, "game": "connect4"}).encode(), "application/json"
        elif path in files:
            name, content_type = files[path]
            body = (WEB / name).read_bytes()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play Connect Four using compiled ReGLU game rules")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the game in your default browser")
    args = parser.parse_args(argv)
    weights = model_bytes()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Connect Four: {url} ({len(weights):,} bytes of compiled weights)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
