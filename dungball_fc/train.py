from __future__ import annotations

import argparse
from pathlib import Path

from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DungBall FC teams through self play")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team-size", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--checkpoint", default="trained-team", help="checkpoint filename without .json")
    args = parser.parse_args()

    trainer = Trainer(seed=args.seed, team_size=args.team_size)
    next_report = 100
    while trainer.completed_rounds < args.episodes:
        trainer.tick()
        if trainer.completed_rounds >= next_report:
            score = trainer.arena.score
            print(
                f"round={trainer.completed_rounds} "
                f"score={score['blue']}:{score['orange']} "
                f"draws={trainer.draws}"
            )
            next_report += 100
    saved = trainer.save_checkpoint(Path(args.checkpoint).stem)
    print(f"checkpoint=checkpoints/{saved}")


if __name__ == "__main__":
    main()
