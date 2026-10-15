"""Check that terrain resume preserves assignments and speed-course progress."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

path = Path(__file__).resolve().parents[2] / "overlay/source/robot_lab/robot_lab/tasks/smalldog/terrain_state.py"
spec = importlib.util.spec_from_file_location("smalldog_terrain_state", path)
state_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state_module)


def make_env(count=4):
    terrain = SimpleNamespace(
        terrain_levels=torch.zeros(count, dtype=torch.long),
        terrain_types=torch.zeros(count, dtype=torch.long),
        terrain_origins=torch.arange(72, dtype=torch.float32).reshape(8, 3, 3),
        env_origins=torch.zeros(count, 3),
    )
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(-0.15, 1.0))))
    speed = SimpleNamespace(count=42, passed=33, error_sum=6.5, completed=40)
    return SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        curriculum_manager=SimpleNamespace(cfg=SimpleNamespace(forward_speed=SimpleNamespace(func=speed))),
        episode_length_buf=torch.full((count,), 15), common_step_counter=100, device="cpu",
    )


class TerrainStateTests(unittest.TestCase):
    def test_stratified_assignments_cover_every_level_in_every_terrain_column(self):
        env = make_env(220)
        env.scene.terrain.terrain_origins = torch.arange(660, dtype=torch.float32).reshape(11, 20, 3)
        levels = state_module.initialize_stratified_terrain(env, 10)
        self.assertEqual(torch.bincount(levels, minlength=11).tolist(), [20] * 11)
        for column in range(20):
            self.assertEqual(levels[env.scene.terrain.terrain_types == column].tolist(), list(range(11)))

    def test_round_trip_preserves_levels_types_origins_and_speed_progress(self):
        source = make_env()
        source.scene.terrain.terrain_levels[:] = torch.tensor([0, 2, 7, 4])
        source.scene.terrain.terrain_types[:] = torch.tensor([2, 1, 0, 2])
        source.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = (-0.15, 1.2)
        saved = state_module.capture_terrain_state(source)
        source.scene.terrain.terrain_levels.zero_()
        target = make_env()
        target.curriculum_manager.cfg.forward_speed.func.count = 0
        state_module.restore_terrain_state(target, saved)
        torch.testing.assert_close(target.scene.terrain.terrain_levels, torch.tensor([0, 2, 7, 4]))
        torch.testing.assert_close(target.scene.terrain.terrain_types, torch.tensor([2, 1, 0, 2]))
        torch.testing.assert_close(
            target.scene.terrain.env_origins,
            target.scene.terrain.terrain_origins[[0, 2, 7, 4], [2, 1, 0, 2]],
        )
        self.assertEqual(target.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x, (-0.15, 1.2))
        self.assertEqual(target.curriculum_manager.cfg.forward_speed.func.count, 42)
        self.assertEqual(target.episode_length_buf.sum(), 0)
        summary = state_module.terrain_state_summary(saved)
        self.assertEqual(summary["level_histogram"], [1, 0, 1, 0, 1, 0, 0, 1])
        self.assertEqual(summary["mean_level"], 3.25)

    def test_incompatible_environment_count_is_reported_before_assignment(self):
        saved = state_module.capture_terrain_state(make_env())
        with self.assertRaisesRegex(ValueError, "same number"):
            state_module.restore_terrain_state(make_env(3), saved)


if __name__ == "__main__":
    unittest.main()
