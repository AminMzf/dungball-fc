from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence


ACTIONS = ("forward", "left", "right", "push")


class Brain(Protocol):
    def act(
        self,
        observation: Sequence[float],
        explore: bool = True,
        forced_action: str | None = None,
    ) -> tuple[str, list[float]]: ...
    def learn(self, reward: float) -> None: ...
    def imitate(self, observation: Sequence[float], action: str, strength: float = 1.0) -> None: ...
    def reset_state(self) -> None: ...


@dataclass
class PlasticBrain:
    """Small softmax policy with dopamine-modulated eligibility traces.

    The original MVP only reinforced the selected action. Positive rewards could
    therefore make *every* frequently selected action more likely, including a
    useless action that happened to dominate at initialization. This version
    uses the policy-gradient term ``chosen - probability`` and a moving reward
    baseline, so rewarded actions win relative to the alternatives.
    """

    inputs: int = 7
    learning_rate: float = 0.025
    trace_decay: float = 0.90
    exploration_rate: float = 0.12
    seed: int = 0
    weights: list[list[float]] = field(init=False)
    eligibility: list[list[float]] = field(init=False)
    _last_observation: list[float] = field(default_factory=list, init=False)
    _last_action: int = field(default=0, init=False)
    reward_baseline: float = field(default=0.0, init=False)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.weights = [[self._rng.uniform(-0.12, 0.12) for _ in range(self.inputs)] for _ in ACTIONS]
        self.eligibility = [[0.0 for _ in range(self.inputs)] for _ in ACTIONS]

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        total = sum(exps)
        return [value / total for value in exps]

    def _policy(self, observation: Sequence[float]) -> tuple[list[float], list[float]]:
        obs = list(observation)[:self.inputs]
        obs.extend([0.0] * (self.inputs - len(obs)))
        logits = [sum(w * x for w, x in zip(row, obs)) for row in self.weights]
        return logits, self._softmax(logits)

    def act(
        self,
        observation: Sequence[float],
        explore: bool = True,
        forced_action: str | None = None,
    ) -> tuple[str, list[float]]:
        obs = list(observation)[:self.inputs]
        obs.extend([0.0] * (self.inputs - len(obs)))
        logits, probabilities = self._policy(obs)
        if forced_action is not None:
            action_idx = ACTIONS.index(forced_action)
        elif explore and self._rng.random() < self.exploration_rate:
            action_idx = self._rng.randrange(len(ACTIONS))
        elif explore:
            draw = self._rng.random()
            cumulative = 0.0
            action_idx = len(ACTIONS) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if draw <= cumulative:
                    action_idx = index
                    break
        else:
            action_idx = max(range(len(logits)), key=logits.__getitem__)
        self._last_observation = obs
        self._last_action = action_idx
        for a in range(len(ACTIONS)):
            for i in range(self.inputs):
                self.eligibility[a][i] *= self.trace_decay
                gradient = (1.0 if a == action_idx else 0.0) - probabilities[a]
                self.eligibility[a][i] += gradient * obs[i]
        scores = [math.tanh(value) for value in logits]
        return ACTIONS[action_idx], scores

    def learn(self, reward: float) -> None:
        advantage = reward - self.reward_baseline
        self.reward_baseline = 0.995 * self.reward_baseline + 0.005 * reward
        dopamine = max(-1.0, min(1.0, advantage))
        for a in range(len(ACTIONS)):
            for i in range(self.inputs):
                self.weights[a][i] = max(-3.0, min(3.0, self.weights[a][i] + self.learning_rate * dopamine * self.eligibility[a][i]))

    def imitate(self, observation: Sequence[float], action: str, strength: float = 1.0) -> None:
        """One supervised coaching update used by the early curriculum."""
        obs = list(observation)[:self.inputs]
        obs.extend([0.0] * (self.inputs - len(obs)))
        _, probabilities = self._policy(obs)
        target = ACTIONS.index(action)
        rate = self.learning_rate * max(0.0, strength)
        for a in range(len(ACTIONS)):
            gradient = (1.0 if a == target else 0.0) - probabilities[a]
            for i, value in enumerate(obs):
                self.weights[a][i] = max(
                    -3.0,
                    min(3.0, self.weights[a][i] + rate * gradient * value),
                )

    def reset_state(self) -> None:
        self.eligibility = [[0.0 for _ in range(self.inputs)] for _ in ACTIONS]

    def ensure_inputs(self, inputs: int) -> None:
        """Expand an older checkpoint for newly added observation features."""
        if inputs <= self.inputs:
            return
        extra = inputs - self.inputs
        for row in self.weights:
            row.extend([0.0] * extra)
        self.inputs = inputs
        self.reset_state()

    def to_dict(self) -> dict:
        return {
            "type": "PlasticBrain",
            "inputs": self.inputs,
            "learningRate": self.learning_rate,
            "traceDecay": self.trace_decay,
            "explorationRate": self.exploration_rate,
            "weights": [row[:] for row in self.weights],
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int = 0) -> "PlasticBrain":
        if data.get("type") != "PlasticBrain":
            raise ValueError("Unsupported brain type")
        inputs = int(data["inputs"])
        weights = data["weights"]
        if len(weights) != len(ACTIONS) or any(len(row) != inputs for row in weights):
            raise ValueError("Checkpoint weight dimensions are invalid")
        brain = cls(
            inputs=inputs,
            learning_rate=float(data.get("learningRate", 0.025)),
            trace_decay=float(data.get("traceDecay", 0.85)),
            exploration_rate=float(data.get("explorationRate", 0.12)),
            seed=seed,
        )
        brain.weights = [[float(value) for value in row] for row in weights]
        brain.reset_state()
        return brain
