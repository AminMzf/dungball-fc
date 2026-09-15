from __future__ import annotations

import math
import random
from dataclasses import dataclass


WIDTH, HEIGHT = 1000.0, 600.0
GOAL_X, GOAL_HALF_HEIGHT = WIDTH, 105.0


@dataclass
class Body:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class DungBallArena:
    def __init__(self, seed: int = 0, max_steps: int = 900):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.episode = 0
        self.total_goals = 0
        self.reset()

    def reset(self) -> list[float]:
        self.episode += 1
        self.step_count = 0
        self.fly = Body(120.0, self.rng.uniform(180.0, 420.0))
        self.ball = Body(self.rng.uniform(360.0, 600.0), self.rng.uniform(150.0, 450.0))
        self.angle = 0.0
        self.previous_ball_goal_distance = self._distance(self.ball.x, self.ball.y, GOAL_X, HEIGHT / 2)
        return self.observe()

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def observe(self) -> list[float]:
        ball_bearing = self._wrap(math.atan2(self.ball.y - self.fly.y, self.ball.x - self.fly.x) - self.angle)
        goal_bearing = self._wrap(math.atan2(HEIGHT / 2 - self.fly.y, GOAL_X - self.fly.x) - self.angle)
        return [
            1.0,
            min(1.0, self._distance(self.fly.x, self.fly.y, self.ball.x, self.ball.y) / WIDTH),
            math.sin(ball_bearing), math.cos(ball_bearing),
            min(1.0, self._distance(self.fly.x, self.fly.y, GOAL_X, HEIGHT / 2) / WIDTH),
            math.sin(goal_bearing), math.cos(goal_bearing),
        ]

    def step(self, action: str) -> tuple[list[float], float, bool, dict]:
        self.step_count += 1
        old_fly_ball = self._distance(self.fly.x, self.fly.y, self.ball.x, self.ball.y)
        if action == "left":
            self.angle -= 0.20
        elif action == "right":
            self.angle += 0.20
        elif action in ("forward", "push"):
            speed = 5.0 if action == "forward" else 6.5
            self.fly.vx += math.cos(self.angle) * speed
            self.fly.vy += math.sin(self.angle) * speed

        self.fly.vx *= 0.78
        self.fly.vy *= 0.78
        self.fly.x = min(WIDTH - 15, max(15, self.fly.x + self.fly.vx))
        self.fly.y = min(HEIGHT - 15, max(15, self.fly.y + self.fly.vy))

        touching = self._distance(self.fly.x, self.fly.y, self.ball.x, self.ball.y) < 38
        if touching:
            force = 6.5 if action == "push" else 2.5
            self.ball.vx += math.cos(self.angle) * force
            self.ball.vy += math.sin(self.angle) * force
        self.ball.vx *= 0.94
        self.ball.vy *= 0.94
        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy
        if self.ball.y < 15 or self.ball.y > HEIGHT - 15:
            self.ball.y = min(HEIGHT - 15, max(15, self.ball.y))
            self.ball.vy *= -0.7
        if self.ball.x < 15:
            self.ball.x = 15
            self.ball.vx *= -0.7

        new_fly_ball = self._distance(self.fly.x, self.fly.y, self.ball.x, self.ball.y)
        new_ball_goal = self._distance(self.ball.x, self.ball.y, GOAL_X, HEIGHT / 2)
        reward = -0.001 + (old_fly_ball - new_fly_ball) * 0.0008
        reward += (self.previous_ball_goal_distance - new_ball_goal) * 0.004
        if touching:
            reward += 0.025
        scored = self.ball.x >= GOAL_X and abs(self.ball.y - HEIGHT / 2) <= GOAL_HALF_HEIGHT
        if scored:
            reward += 1.0
            self.total_goals += 1
        done = scored or self.step_count >= self.max_steps or self.ball.x > WIDTH + 30
        self.previous_ball_goal_distance = new_ball_goal
        return self.observe(), reward, done, {"scored": scored, "touching": touching}

    def snapshot(self) -> dict:
        return {
            "episode": self.episode,
            "step": self.step_count,
            "goals": self.total_goals,
            "field": {"width": WIDTH, "height": HEIGHT, "goalHalfHeight": GOAL_HALF_HEIGHT},
            "fly": {"x": self.fly.x, "y": self.fly.y, "angle": self.angle},
            "ball": {"x": self.ball.x, "y": self.ball.y},
        }
