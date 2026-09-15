from __future__ import annotations

import argparse

from .brain import PlasticBrain
from .environment import DungBallArena


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    arena, brain = DungBallArena(args.seed), PlasticBrain(seed=args.seed)
    goals, rewards = 0, 0.0
    for episode in range(1, args.episodes + 1):
        done = False
        while not done:
            action, _ = brain.act(arena.observe())
            _, reward, done, info = arena.step(action)
            brain.learn(reward)
            rewards += reward
            goals += int(info["scored"])
        arena.reset()
        brain.reset_state()
        if episode % 100 == 0:
            print(f"episode={episode} goals={goals} cumulative_reward={rewards:.3f}")


if __name__ == "__main__":
    main()
