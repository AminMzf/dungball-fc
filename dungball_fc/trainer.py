from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from .brain import ACTIONS, PlasticBrain
from .environment import DungBallArena


@dataclass
class Trainer:
    seed: int = 0
    arena: DungBallArena = field(init=False)
    brain: PlasticBrain = field(init=False)
    training: bool = True
    paused: bool = False
    last_reward: float = 0.0
    last_action: str = "forward"
    scores: list[float] = field(default_factory=lambda: [0.0] * len(ACTIONS))
    recent_rewards: list[float] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.arena = DungBallArena(self.seed)
        self.brain = PlasticBrain(seed=self.seed)

    def tick(self) -> None:
        with self.lock:
            if self.paused:
                return
            observation = self.arena.observe()
            self.last_action, self.scores = self.brain.act(observation, explore=self.training)
            _, self.last_reward, done, _ = self.arena.step(self.last_action)
            if self.training:
                self.brain.learn(self.last_reward)
            self.recent_rewards.append(self.last_reward)
            self.recent_rewards = self.recent_rewards[-300:]
            if done:
                self.arena.reset()
                self.brain.reset_state()

    def reset_brain(self) -> None:
        with self.lock:
            self.brain = PlasticBrain(seed=self.seed)
            self.arena = DungBallArena(self.seed)
            self.recent_rewards.clear()

    def snapshot(self) -> dict:
        with self.lock:
            result = self.arena.snapshot()
            result.update({
                "training": self.training,
                "paused": self.paused,
                "action": self.last_action,
                "reward": self.last_reward,
                "dopamine": max(-1.0, min(1.0, self.last_reward)),
                "meanReward": sum(self.recent_rewards) / max(1, len(self.recent_rewards)),
                "activity": dict(zip(ACTIONS, self.scores)),
            })
            return result
