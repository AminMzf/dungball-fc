from __future__ import annotations

import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .trainer import Trainer


ROOT = Path(__file__).resolve().parent.parent
trainer = Trainer(seed=42)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def _json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/state":
            self._json(trainer.snapshot())
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/pause":
            trainer.paused = not trainer.paused
        elif path == "/api/mode":
            trainer.training = not trainer.training
        elif path == "/api/reset":
            trainer.reset_brain()
        else:
            self._json({"error": "not found"}, 404)
            return
        self._json(trainer.snapshot())

    def log_message(self, format: str, *args) -> None:
        return


def training_loop() -> None:
    while True:
        trainer.tick()
        time.sleep(1 / 60)


def main() -> None:
    threading.Thread(target=training_loop, daemon=True).start()
    print("DungBall FC running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
