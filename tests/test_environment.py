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

    def test_team_observation_includes_near_ball_mode_features(self):
        arena = TeamArena(seed=1, team_size=2)
        fly = arena.flies[0]
        arena.ball.x, arena.ball.y = fly.x + 20, fly.y
        observation = arena.observe(fly)
        self.assertEqual(len(observation), 23)
        self.assertGreater(observation[15], 0.5)

    def test_coach_turns_toward_ball_and_pushes_when_aligned(self):
        arena = TeamArena(seed=1, team_size=2)
        fly = arena.flies[0]
        fly.x, fly.y, fly.angle = 450, 300, 0
        arena.ball.x, arena.ball.y = 480, 300
        self.assertEqual(arena.coach_action(fly), "push")
        fly.angle = -1.0
        self.assertEqual(arena.coach_action(fly), "right")

    def test_imitation_increases_probability_of_coached_action(self):
        brain = PlasticBrain(inputs=7, seed=1)
        observation = [1.0, 0.3, 0.0, 1.0, 0.8, 0.0, 1.0]
        before = brain._policy(observation)[1][ACTIONS.index("forward")]
        for _ in range(20):
            brain.imitate(observation, "forward")
        after = brain._policy(observation)[1][ACTIONS.index("forward")]
        self.assertGreater(after, before)

    def test_old_brain_can_expand_for_new_observations(self):
        brain = PlasticBrain(inputs=15, seed=1)
        original = [row[:] for row in brain.weights]
        brain.ensure_inputs(23)
        self.assertEqual(brain.inputs, 23)
        self.assertTrue(all(row[:15] == old for row, old in zip(brain.weights, original)))
        self.assertTrue(all(row[15:] == [0.0] * 8 for row in brain.weights))

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

    def test_frozen_evaluation_reports_movement_metrics(self):
        trainer = Trainer(seed=3)
        result = trainer.evaluate(rounds=1)
        self.assertEqual(result["rounds"], 1)
        self.assertIn("goalRate", result)
        self.assertGreaterEqual(result["ballTravelPerRound"], 0.0)


if __name__ == "__main__":
    unittest.main()
