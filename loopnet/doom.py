"""Serve a tiny first-person shooter whose simulation and pixels are neural."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit
import webbrowser

from .structured import compile_structured, lower
from .language import parse
from .browser import browser_program, manifest_bytes


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "doom.flow"
WEB = Path(__file__).with_name("web")
WIDTH, HEIGHT = 64, 48
VIEW_HEIGHT = 40
INITIAL = {"player_x": 4, "player_y": 5.7, "dir_x": 0, "dir_y": -1,
           "ammo": 60, "cooldown": 0, "alive[0]": 1, "alive[1]": 1, "alive[2]": 1}
PALETTE = [[6, 9, 13], [15, 19, 27], [37, 30, 31], [48, 42, 43],
           [24, 48, 48], [32, 69, 68], [42, 89, 84], [55, 110, 99],
           [76, 134, 111], [104, 158, 126], [137, 181, 143], [174, 202, 162],
           [109, 24, 30], [177, 39, 43], [229, 68, 52], [255, 119, 75],
           [35, 40, 46], [87, 91, 97], [156, 166, 171], [203, 236, 190], [255, 224, 141],
           [22, 29, 27], [70, 90, 66], [245, 191, 105], [229, 104, 85], [166, 255, 112]]

STATIC_FILES = {"index.html": "doom/index.html", "app.mjs": "doom/app.mjs",
                "style.css": "doom/style.css", "game-worker.mjs": "doom/game-worker.mjs",
                "atlas-viewer.mjs": "doom/atlas-viewer.mjs", "neural.mjs": "neural.mjs",
                "compiler.mjs": "compiler.mjs", "activation-atlas.mjs": "activation-atlas.mjs"}


@lru_cache(maxsize=1)
def browser_assets():
    program = parse(lower(SOURCE.read_text(), max_unrolled_iterations=80000))
    manifest, code = browser_program(program)
    manifest["game"] = {"width": WIDTH, "height": HEIGHT, "initial": INITIAL, "palette": PALETTE}
    return manifest_bytes(manifest), code


def export_site(directory):
    """Write a self-contained GitHub Pages site with source, not neural weights."""
    manifest, code = browser_assets()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in STATIC_FILES.items():
        shutil.copyfile(WEB / source, directory / name)
    (directory / "program.json").write_bytes(manifest)
    (directory / "program.bin.gz").write_bytes(code)
    (directory / "doom.flow.txt").write_bytes(SOURCE.read_bytes())
    (directory / ".nojekyll").write_text("")
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


@lru_cache(maxsize=1)
def compiled_game():
    return compile_structured(SOURCE.read_text(), max_unrolled_iterations=80000)


@lru_cache(maxsize=1)
def model_bytes():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "doom.json"
        compiled_game().save(path)
        return path.read_bytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        files = {"/": ("doom/index.html", "text/html"),
                 "/activation-trace.mjs": ("activation-trace.mjs", "text/javascript"),
                 "/activation-viewer.mjs": ("doom/activation-viewer.mjs", "text/javascript")}
        files.update({"/" + name: (source, "text/html" if name.endswith(".html") else
                                  "text/css" if name.endswith(".css") else "text/javascript")
                      for name, source in STATIC_FILES.items()})
        if path == "/api/network":
            body, content_type = model_bytes(), "application/json"
        elif path in {"/program.json", "/program.bin.gz"}:
            assets = browser_assets()
            body, content_type = ((assets[0], "application/json") if path.endswith(".json")
                                  else (assets[1], "application/gzip"))
        elif path == "/api/config":
            model = compiled_game()
            body = json.dumps({"width": WIDTH, "height": HEIGHT, "initial": INITIAL,
                               "palette": PALETTE, "summary": model.summary()}).encode()
            content_type = "application/json"
        elif path in {"/source", "/doom.flow", "/doom.flow.txt"}:
            body, content_type = SOURCE.read_bytes(), "text/plain"
        elif path == "/health":
            body, content_type = b'{"ok":true,"game":"neural-doom"}', "application/json"
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
    parser = argparse.ArgumentParser(description="Play a 64x48 shooter compiled into a recurrent neural network")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--export", type=Path, metavar="DIRECTORY",
                        help="export a static site for GitHub Pages; visitors generate the network locally")
    args = parser.parse_args(argv)
    print("Preparing browser source program…", flush=True)
    if args.export:
        size = export_site(args.export)
        print(f"Static Doom: {args.export.resolve()} ({size:,} bytes; no bundled weights)", flush=True)
        return
    manifest, code = browser_assets()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Neural Doom: {url} ({len(manifest) + len(code):,} bytes of source; network built in browser)", flush=True)
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
