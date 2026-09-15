from __future__ import annotations

import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .trainer import Trainer


ROOT = Path(__file__).resolve().parent.parent
trainer = Trainer(seed=42, team_size=2)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def _json(self, data: dict | list, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        length = min(int(self.headers.get("Content-Length", 0)), 5_000_000)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(trainer.snapshot())
            return
        if path == "/api/checkpoints":
            self._json(trainer.list_checkpoints())
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/pause":
                trainer.paused = not trainer.paused
            elif path == "/api/mode":
                trainer.training = not trainer.training
            elif path == "/api/reset":
                trainer.reset_brain()
            elif path == "/api/team-size":
                trainer.set_team_size(int(self._body().get("teamSize", 2)))
            elif path == "/api/checkpoint/save":
                name = trainer.save_checkpoint(self._body().get("name", "latest"))
                self._json({"ok": True, "name": name})
                return
            elif path == "/api/checkpoint/load":
                body = self._body()
                trainer.load_checkpoint(body["name"], bool(body.get("evaluate", True)))
            elif path == "/api/checkpoint/import":
                name = parse_qs(parsed.query).get("name", ["imported"])[0]
                saved = trainer.import_checkpoint(self._body(), Path(name).stem)
                self._json({"ok": True, "name": saved})
                return
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json(trainer.snapshot())
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format: str, *args) -> None:
        return


def training_loop() -> None:
    while True:
        trainer.tick()
        time.sleep(1 / 60)


def main() -> None:
    threading.Thread(target=training_loop, daemon=True).start()
    print("DungBall FC 3D running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
