import unittest
from pathlib import Path
from unittest.mock import patch

from dungball_fc.brain import ACTIONS, PlasticBrain
from dungball_fc.environment import DungBallArena, GOAL_X, HEIGHT, TeamArena
from dungball_fc.trainer import Trainer


class ArenaTests(unittest.TestCase):
    def test_observation_shape_and_range(self):
        arena = DungBallArena(seed=1)
        obs = arena.observe()
        self.assertEqual(len(obs), 7)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in obs))

    def test_goal_scores_and_ends_episode(self):
        arena = DungBallArena(seed=1)
        arena.ball.x, arena.ball.y, arena.ball.vx = GOAL_X - 1, HEIGHT / 2, 5
        _, reward, done, info = arena.step("left")
        self.assertTrue(done)
        self.assertTrue(info["scored"])
        self.assertGreater(reward, 0.9)

    def test_positive_dopamine_changes_weights(self):
        brain = PlasticBrain(seed=1)
        before = [row[:] for row in brain.weights]
        action, _ = brain.act([1.0] * 7, explore=False)
        brain.learn(1.0)
        action_index = ACTIONS.index(action)
        self.assertNotEqual(before[action_index], brain.weights[action_index])

    def test_team_arena_starts_as_two_vs_two(self):
        arena = TeamArena(seed=1, team_size=2)
        self.assertEqual(len(arena.flies), 4)
        self.assertEqual({fly.team for fly in arena.flies}, {"blue", "orange"})
        observations, rewards, done, _ = arena.step({fly.id: "forward" for fly in arena.flies})
        self.assertEqual(set(observations), {fly.id for fly in arena.flies})
        self.assertEqual(set(rewards), {"blue", "orange"})
        self.assertFalse(done)

    def test_team_arena_scoring_works_from_both_sides(self):
        arena = TeamArena(seed=1, team_size=2)
        arena.ball.x, arena.ball.y, arena.ball.vx = 999, HEIGHT / 2, 5
        _, rewards, done, info = arena.step({})
        self.assertTrue(done)
        self.assertEqual(info["scoringTeam"], "blue")
        self.assertGreater(rewards["blue"], 0.9)

    def test_checkpoint_round_trip_enters_evaluation(self):
        trainer = Trainer(seed=3, checkpoint_dir=Path("checkpoints"))
        for _ in range(4):
            trainer.tick()
        before = [row[:] for row in trainer.brains["blue"].weights]
        with patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write:
            name = trainer.save_checkpoint("unit-test")
        checkpoint_json = write.call_args.args[0]
        trainer.reset_brain()
        with patch.object(Path, "read_text", return_value=checkpoint_json):
            trainer.load_checkpoint(name, evaluate=True)
        self.assertEqual(before, trainer.brains["blue"].weights)
        self.assertFalse(trainer.training)
        self.assertEqual(trainer.active_checkpoint, "unit-test.json")


if __name__ == "__main__":
    unittest.main()
