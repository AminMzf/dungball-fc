from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .brain import ACTIONS, PlasticBrain
from .environment import TeamArena


CHECKPOINT_VERSION = 1


@dataclass
class Trainer:
    seed: int = 0
    team_size: int = 2
    checkpoint_dir: Path | None = None
    arena: TeamArena = field(init=False)
    brains: dict[str, PlasticBrain] = field(init=False)
    training: bool = True
    paused: bool = False
    last_rewards: dict[str, float] = field(default_factory=lambda: {"blue": 0.0, "orange": 0.0})
    last_actions: dict[str, str] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)
    recent_rewards: dict[str, list[float]] = field(default_factory=lambda: {"blue": [], "orange": []})
    completed_rounds: int = 0
    draws: int = 0
    active_checkpoint: str | None = None
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.arena = TeamArena(self.seed, self.team_size)
        self.brains = {
            "blue": PlasticBrain(inputs=15, seed=self.seed),
            "orange": PlasticBrain(inputs=15, seed=self.seed + 1),
        }
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"

    def tick(self) -> None:
        with self.lock:
            if self.paused:
                return
            actions: dict[str, str] = {}
            for fly in self.arena.flies:
                action, values = self.brains[fly.team].act(
                    self.arena.observe(fly), explore=self.training
                )
                actions[fly.id] = action
                self.scores[fly.id] = values
            self.last_actions = actions
            _, self.last_rewards, done, _ = self.arena.step(actions)
            if self.training:
                for team, brain in self.brains.items():
                    brain.learn(self.last_rewards[team])
            for team in ("blue", "orange"):
                self.recent_rewards[team].append(self.last_rewards[team])
                self.recent_rewards[team] = self.recent_rewards[team][-600:]
            if done:
                self.completed_rounds += 1
                if self.arena.step_count >= self.arena.max_steps:
                    self.draws += 1
                self.arena.reset()
                for brain in self.brains.values():
                    brain.reset_state()

    def reset_brain(self) -> None:
        with self.lock:
            self.brains = {
                "blue": PlasticBrain(inputs=15, seed=self.seed),
                "orange": PlasticBrain(inputs=15, seed=self.seed + 1),
            }
            self.arena = TeamArena(self.seed, self.team_size)
            self.recent_rewards = {"blue": [], "orange": []}
            self.completed_rounds = 0
            self.draws = 0
            self.active_checkpoint = None
            self.training = True

    def set_team_size(self, team_size: int) -> None:
        with self.lock:
            self.team_size = max(1, min(5, int(team_size)))
            self.arena.set_team_size(self.team_size)

    @staticmethod
    def _safe_name(name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip(".-")
        return (clean or "checkpoint")[:80]

    def checkpoint_data(self) -> dict:
        with self.lock:
            return self._checkpoint_data_unlocked()

    def _checkpoint_data_unlocked(self) -> dict:
        return {
            "format": "dungball-fc.checkpoint",
            "version": CHECKPOINT_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "teamSize": self.team_size,
            "episode": self.arena.round,
            "score": dict(self.arena.score),
            "completedRounds": self.completed_rounds,
            "brains": {team: brain.to_dict() for team, brain in self.brains.items()},
        }

    def save_checkpoint(self, name: str = "latest") -> str:
        safe_name = self._safe_name(name)
        path = self.checkpoint_dir / f"{safe_name}.json"
        with self.lock:
            data = self._checkpoint_data_unlocked()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.active_checkpoint = path.name
        return path.name

    def import_checkpoint(self, data: dict, name: str = "imported") -> str:
        self._validate_checkpoint(data)
        safe_name = self._safe_name(name)
        path = self.checkpoint_dir / f"{safe_name}.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path.name

    def load_checkpoint(self, name: str, evaluate: bool = True) -> None:
        safe_name = self._safe_name(Path(name).stem)
        path = self.checkpoint_dir / f"{safe_name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self._validate_checkpoint(data)
        with self.lock:
            self.brains = {
                team: PlasticBrain.from_dict(data["brains"][team], self.seed + index)
                for index, team in enumerate(("blue", "orange"))
            }
            self.team_size = max(1, min(5, int(data.get("teamSize", self.team_size))))
            self.arena = TeamArena(self.seed, self.team_size)
            self.recent_rewards = {"blue": [], "orange": []}
            self.completed_rounds = 0
            self.draws = 0
            self.active_checkpoint = path.name
            self.training = not evaluate
            self.paused = False

    @staticmethod
    def _validate_checkpoint(data: dict) -> None:
        if data.get("format") != "dungball-fc.checkpoint":
            raise ValueError("This is not a DungBall FC checkpoint")
        if int(data.get("version", 0)) != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        if set(data.get("brains", {})) != {"blue", "orange"}:
            raise ValueError("Checkpoint must contain blue and orange brains")
        for index, team in enumerate(("blue", "orange")):
            PlasticBrain.from_dict(data["brains"][team], index)

    def list_checkpoints(self) -> list[dict]:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for path in sorted(self.checkpoint_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._validate_checkpoint(data)
                results.append({
                    "name": path.name,
                    "createdAt": data.get("createdAt"),
                    "episode": data.get("episode", 0),
                    "teamSize": data.get("teamSize", 2),
                    "active": path.name == self.active_checkpoint,
                })
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return results

    def snapshot(self) -> dict:
        with self.lock:
            result = self.arena.snapshot()
            total_possession = sum(self.arena.possession.values())
            activity = {
                fly_id: dict(zip(ACTIONS, values))
                for fly_id, values in self.scores.items()
            }
            result.update({
                "training": self.training,
                "paused": self.paused,
                "actions": dict(self.last_actions),
                "rewards": dict(self.last_rewards),
                "dopamine": {team: max(-1.0, min(1.0, value)) for team, value in self.last_rewards.items()},
                "meanReward": {
                    team: sum(values) / max(1, len(values))
                    for team, values in self.recent_rewards.items()
                },
                "activity": activity,
                "completedRounds": self.completed_rounds,
                "draws": self.draws,
                "activeCheckpoint": self.active_checkpoint,
                "possessionPercent": {
                    team: round(value * 100 / max(1, total_possession))
                    for team, value in self.arena.possession.items()
                },
            })
            return result
