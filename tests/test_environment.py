import unittest

from dungball_fc.brain import ACTIONS, PlasticBrain
from dungball_fc.environment import DungBallArena, GOAL_X, HEIGHT


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


if __name__ == "__main__":
    unittest.main()
