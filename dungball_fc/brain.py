from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence


ACTIONS = ("forward", "left", "right", "push")


class Brain(Protocol):
    def act(self, observation: Sequence[float], explore: bool = True) -> tuple[str, list[float]]: ...
    def learn(self, reward: float) -> None: ...
    def reset_state(self) -> None: ...


@dataclass
class PlasticBrain:
    """Tiny dopamine-modulated policy used as an MVP seam for MaleCNS."""

    inputs: int = 7
    learning_rate: float = 0.025
    trace_decay: float = 0.85
    seed: int = 0
    weights: list[list[float]] = field(init=False)
    eligibility: list[list[float]] = field(init=False)
    _last_observation: list[float] = field(default_factory=list, init=False)
    _last_action: int = field(default=0, init=False)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.weights = [[self._rng.uniform(-0.12, 0.12) for _ in range(self.inputs)] for _ in ACTIONS]
        self.eligibility = [[0.0 for _ in range(self.inputs)] for _ in ACTIONS]

    def act(self, observation: Sequence[float], explore: bool = True) -> tuple[str, list[float]]:
        obs = list(observation)
        scores = [math.tanh(sum(w * x for w, x in zip(row, obs))) for row in self.weights]
        if explore and self._rng.random() < 0.12:
            action_idx = self._rng.randrange(len(ACTIONS))
        else:
            action_idx = max(range(len(scores)), key=scores.__getitem__)
        self._last_observation = obs
        self._last_action = action_idx
        for a in range(len(ACTIONS)):
            for i in range(self.inputs):
                self.eligibility[a][i] *= self.trace_decay
        for i, value in enumerate(obs):
            self.eligibility[action_idx][i] += value
        return ACTIONS[action_idx], scores

    def learn(self, reward: float) -> None:
        dopamine = max(-1.0, min(1.0, reward))
        for a in range(len(ACTIONS)):
            for i in range(self.inputs):
                self.weights[a][i] = max(-3.0, min(3.0, self.weights[a][i] + self.learning_rate * dopamine * self.eligibility[a][i]))

    def reset_state(self) -> None:
        self.eligibility = [[0.0 for _ in range(self.inputs)] for _ in ACTIONS]
