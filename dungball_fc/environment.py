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


@dataclass
class FlyBody(Body):
    """A player in the team arena."""

    id: str = ""
    team: str = "blue"
    number: int = 1
    angle: float = 0.0


class TeamArena:
    """Symmetric dung-ball football for 1v1 through 5v5 self play."""

    def __init__(self, seed: int = 0, team_size: int = 2, max_steps: int = 1200):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.team_size = max(1, min(5, team_size))
        self.round = 0
        self.score = {"blue": 0, "orange": 0}
        self.possession = {"blue": 0, "orange": 0}
        self.reset()

    def set_team_size(self, team_size: int) -> None:
        self.team_size = max(1, min(5, int(team_size)))
        self.score = {"blue": 0, "orange": 0}
        self.round = 0
        self.reset()

    def reset(self) -> dict[str, list[float]]:
        self.round += 1
        self.step_count = 0
        self.last_touch = None
        self.flies: list[FlyBody] = []
        spacing = HEIGHT / (self.team_size + 1)
        for team in ("blue", "orange"):
            for index in range(self.team_size):
                lane = spacing * (index + 1)
                stagger = 36.0 if index % 2 else 0.0
                if team == "blue":
                    x, angle = 190.0 + stagger, 0.0
                else:
                    x, angle = WIDTH - 190.0 - stagger, math.pi
                self.flies.append(FlyBody(
                    x=x,
                    y=lane,
                    id=f"{team}-{index + 1}",
                    team=team,
                    number=index + 1,
                    angle=angle,
                ))
        self.ball = Body(WIDTH / 2, HEIGHT / 2 + self.rng.uniform(-55.0, 55.0))
        return {fly.id: self.observe(fly) for fly in self.flies}

    @staticmethod
    def _distance(a: Body, b: Body) -> float:
        return math.hypot(b.x - a.x, b.y - a.y)

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _relative(self, fly: FlyBody, target: Body) -> tuple[float, float, float]:
        bearing = self._wrap(math.atan2(target.y - fly.y, target.x - fly.x) - fly.angle)
        return min(1.0, self._distance(fly, target) / WIDTH), math.sin(bearing), math.cos(bearing)

    def _navigation_bearing(self, fly: FlyBody) -> tuple[float, float]:
        """Return distance to ball and bearing for a useful ball approach."""
        attack_direction = 1.0 if fly.team == "blue" else -1.0
        ball_distance = self._distance(fly, self.ball)
        if ball_distance > 62.0:
            target = Body(
                min(WIDTH - 25.0, max(25.0, self.ball.x - attack_direction * 46.0)),
                self.ball.y,
            )
        else:
            target = Body(WIDTH if fly.team == "blue" else 0.0, HEIGHT / 2)
        bearing = self._wrap(math.atan2(target.y - fly.y, target.x - fly.x) - fly.angle)
        return ball_distance, bearing

    def observe(self, fly: FlyBody) -> list[float]:
        enemy_x = WIDTH if fly.team == "blue" else 0.0
        enemy_goal = Body(enemy_x, HEIGHT / 2)
        teammates = [other for other in self.flies if other.team == fly.team and other.id != fly.id]
        opponents = [other for other in self.flies if other.team != fly.team]
        teammate = min(teammates, key=lambda other: self._distance(fly, other), default=fly)
        opponent = min(opponents, key=lambda other: self._distance(fly, other), default=fly)
        team_direction = 1.0 if fly.team == "blue" else -1.0
        ball = self._relative(fly, self.ball)
        goal = self._relative(fly, enemy_goal)
        # These gated features let a linear policy express the two useful modes:
        # chase the ball while far away, then face the goal while in possession.
        # Without the interaction terms it cannot represent that switch.
        near_ball = max(0.0, min(1.0, 1.0 - ball[0] / 0.12))
        far_ball = 1.0 - near_ball
        ball_distance, navigation_bearing = self._navigation_bearing(fly)
        push_ready = float(ball_distance < 44.0 and abs(navigation_bearing) <= 0.16)
        return [
            1.0,
            *ball,
            *goal,
            *self._relative(fly, teammate),
            *self._relative(fly, opponent),
            max(-1.0, min(1.0, self.ball.vx * team_direction / 12.0)),
            max(-1.0, min(1.0, self.ball.vy / 12.0)),
            near_ball,
            near_ball * goal[1],
            near_ball * goal[2],
            far_ball * ball[1],
            far_ball * ball[2],
            math.sin(navigation_bearing),
            math.cos(navigation_bearing),
            push_ready,
        ]

    def coach_action(self, fly: FlyBody) -> str:
        """A simple curriculum teacher: get behind the ball, then push at goal."""
        ball_distance, bearing = self._navigation_bearing(fly)
        if bearing > 0.16:
            return "right"
        if bearing < -0.16:
            return "left"
        return "push" if ball_distance < 44.0 else "forward"

    def step(self, actions: dict[str, str]) -> tuple[dict[str, list[float]], dict[str, float], bool, dict]:
        self.step_count += 1
        old_ball_x = self.ball.x
        old_ball_distances = {
            team: min(self._distance(fly, self.ball) for fly in self.flies if fly.team == team)
            for team in ("blue", "orange")
        }
        contacts = {"blue": 0, "orange": 0}

        for fly in self.flies:
            action = actions.get(fly.id, "forward")
            if action == "left":
                fly.angle -= 0.18
            elif action == "right":
                fly.angle += 0.18
            elif action in ("forward", "push"):
                speed = 4.6 if action == "forward" else 6.0
                fly.vx += math.cos(fly.angle) * speed
                fly.vy += math.sin(fly.angle) * speed
            fly.vx *= 0.76
            fly.vy *= 0.76
            fly.x = min(WIDTH - 22.0, max(22.0, fly.x + fly.vx))
            fly.y = min(HEIGHT - 22.0, max(22.0, fly.y + fly.vy))

        # Keep players from occupying exactly the same space.
        for index, fly in enumerate(self.flies):
            for other in self.flies[index + 1:]:
                dx, dy = other.x - fly.x, other.y - fly.y
                distance = max(0.001, math.hypot(dx, dy))
                if distance < 30.0:
                    push = (30.0 - distance) * 0.12
                    nx, ny = dx / distance, dy / distance
                    fly.x -= nx * push
                    fly.y -= ny * push
                    other.x += nx * push
                    other.y += ny * push

        for fly in self.flies:
            distance = self._distance(fly, self.ball)
            if distance < 38.0:
                action = actions.get(fly.id, "forward")
                force = 6.6 if action == "push" else 2.4
                self.ball.vx += math.cos(fly.angle) * force + fly.vx * 0.22
                self.ball.vy += math.sin(fly.angle) * force + fly.vy * 0.22
                contacts[fly.team] += 1
                self.possession[fly.team] += 1
                self.last_touch = fly.id

        self.ball.vx *= 0.945
        self.ball.vy *= 0.945
        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy
        if self.ball.y < 18.0 or self.ball.y > HEIGHT - 18.0:
            self.ball.y = min(HEIGHT - 18.0, max(18.0, self.ball.y))
            self.ball.vy *= -0.76

        in_goal = abs(self.ball.y - HEIGHT / 2) <= GOAL_HALF_HEIGHT
        scoring_team = None
        if self.ball.x >= WIDTH and in_goal:
            scoring_team = "blue"
        elif self.ball.x <= 0 and in_goal:
            scoring_team = "orange"
        elif self.ball.x < 18.0 or self.ball.x > WIDTH - 18.0:
            self.ball.x = min(WIDTH - 18.0, max(18.0, self.ball.x))
            self.ball.vx *= -0.76

        progress = (self.ball.x - old_ball_x) * 0.003
        approach = {
            team: old_ball_distances[team] - min(
                self._distance(fly, self.ball) for fly in self.flies if fly.team == team
            )
            for team in ("blue", "orange")
        }
        rewards = {
            "blue": -0.0005 + progress + approach["blue"] * 0.001,
            "orange": -0.0005 - progress + approach["orange"] * 0.001,
        }
        if scoring_team:
            other = "orange" if scoring_team == "blue" else "blue"
            self.score[scoring_team] += 1
            rewards[scoring_team] += 1.0
            rewards[other] -= 1.0

        done = scoring_team is not None or self.step_count >= self.max_steps
        observations = {fly.id: self.observe(fly) for fly in self.flies}
        return observations, rewards, done, {
            "scoringTeam": scoring_team,
            "contacts": contacts,
            "lastTouch": self.last_touch,
        }

    def snapshot(self) -> dict:
        return {
            "episode": self.round,
            "step": self.step_count,
            "matchProgress": self.step_count / self.max_steps,
            "field": {"width": WIDTH, "height": HEIGHT, "goalHalfHeight": GOAL_HALF_HEIGHT},
            "teamSize": self.team_size,
            "score": dict(self.score),
            "possession": dict(self.possession),
            "lastTouch": self.last_touch,
            "flies": [
                {
                    "id": fly.id,
                    "team": fly.team,
                    "number": fly.number,
                    "x": fly.x,
                    "y": fly.y,
                    "angle": fly.angle,
                    "speed": math.hypot(fly.vx, fly.vy),
                }
                for fly in self.flies
            ],
            "ball": {"x": self.ball.x, "y": self.ball.y, "vx": self.ball.vx, "vy": self.ball.vy},
        }
