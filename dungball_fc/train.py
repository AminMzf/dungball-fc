from __future__ import annotations

import argparse
from pathlib import Path

from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DungBall FC teams through self play")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--steps",
        type=int,
        help="environment steps to train (recommended; default: 1,000,000)",
    )
    duration.add_argument(
        "--episodes",
        type=int,
        help="completed rounds to train (legacy; a round can take 1,200 steps)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team-size", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--coach-steps", type=int, default=75_000)
    parser.add_argument("--report-every", type=int, default=25_000)
    parser.add_argument("--eval-rounds", type=int, default=20)
    parser.add_argument("--resume", help="checkpoint name to continue training")
    parser.add_argument("--checkpoint", default="trained-team", help="checkpoint filename without .json")
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.coach_steps < 0:
        parser.error("--coach-steps cannot be negative")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if args.eval_rounds <= 0:
        parser.error("--eval-rounds must be positive")

    trainer = Trainer(seed=args.seed, team_size=args.team_size, coach_steps=args.coach_steps)
    if args.resume:
        trainer.load_checkpoint(args.resume, evaluate=False)

    requested_steps = args.steps if args.steps is not None else (None if args.episodes is not None else 1_000_000)
    target_steps = trainer.total_steps + requested_steps if requested_steps is not None else None
    target_rounds = trainer.completed_rounds + args.episodes if args.episodes is not None else None
    next_report = trainer.total_steps + args.report_every
    while (
        (target_steps is not None and trainer.total_steps < target_steps)
        or (target_rounds is not None and trainer.completed_rounds < target_rounds)
    ):
        trainer.tick()
        if trainer.total_steps >= next_report:
            score = trainer.arena.score
            print(
                f"step={trainer.total_steps} "
                f"round={trainer.completed_rounds} "
                f"score={score['blue']}:{score['orange']} "
                f"draws={trainer.draws} "
                f"coach={trainer.coach_probability:.1%}"
            )
            next_report += args.report_every
    evaluation = trainer.evaluate(args.eval_rounds)
    print(
        f"evaluation_rounds={evaluation['rounds']} "
        f"goals={evaluation['goals']['blue']}:{evaluation['goals']['orange']} "
        f"goal_rate={evaluation['goalRate']:.1%} "
        f"contacts_per_round={evaluation['contactsPerRound']:.1f} "
        f"ball_travel_per_round={evaluation['ballTravelPerRound']:.1f} "
        f"mean_round_steps={evaluation['meanRoundSteps']:.1f}"
    )
    saved = trainer.save_checkpoint(Path(args.checkpoint).stem)
    print(f"checkpoint=checkpoints/{saved}")


if __name__ == "__main__":
    main()
