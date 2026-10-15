"""Pure tensor checks for the SmallDog R4 gait quality terms."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "overlay/source/robot_lab/robot_lab/tasks/smalldog"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_refinement_mdp():
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = object
    math_module = types.ModuleType("isaaclab.utils.math")
    math_module.quat_apply = lambda quaternion, vector: vector
    warp_module = types.ModuleType("isaaclab.utils.warp")
    warp_module.raycast_mesh = lambda *args, **kwargs: None
    for name, module in {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.managers": managers,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math_module,
        "isaaclab.utils.warp": warp_module,
        "robot_lab": types.ModuleType("robot_lab"),
        "robot_lab.tasks": types.ModuleType("robot_lab.tasks"),
        "robot_lab.tasks.smalldog": types.ModuleType("robot_lab.tasks.smalldog"),
    }.items():
        sys.modules[name] = module
    _load_module("robot_lab.tasks.smalldog.gait_mdp", PACKAGE_DIR / "gait_mdp.py")
    return _load_module(
        "robot_lab.tasks.smalldog.refinement_mdp", PACKAGE_DIR / "refinement_mdp.py"
    )


mdp = _load_refinement_mdp()


class CommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, _name: str) -> torch.Tensor:
        return self.command


def fake_env(command: torch.Tensor, episode_steps: torch.Tensor | None = None):
    count = command.shape[0]
    if episode_steps is None:
        episode_steps = torch.zeros(count)
    return SimpleNamespace(
        command_manager=CommandManager(command),
        cfg=SimpleNamespace(
            gait_frequency_hz=1.5,
            gait_frequency_max_hz=2.5,
            gait_frequency_speed_range=(0.3, 1.0),
            gait_stance_fraction=0.60,
            gait_clearance_m=0.04,
        ),
        episode_length_buf=episode_steps,
        step_dt=0.02,
        device=torch.device("cpu"),
    )


class RefinementMdpTests(unittest.TestCase):
    def test_foot_indices_preserve_order_without_repeated_sensor_metadata(self):
        gait = sys.modules["robot_lab.tasks.smalldog.gait_mdp"]

        class Sensor:
            def __init__(self, names):
                self.names = names
                self.metadata_reads = 0

            @property
            def body_names(self):
                self.metadata_reads += 1
                return self.names

        class Scene(dict):
            pass

        asset = SimpleNamespace(find_bodies=lambda names, preserve_order: ([3, 1, 4, 2], list(names)))
        names = ["base", *reversed(gait.FOOT_NAMES)]
        sensor = Sensor(names)
        scene = Scene(robot=asset)
        scene.sensors = {"contact": sensor}
        env = SimpleNamespace(scene=scene)
        asset_cfg = SimpleNamespace(name="robot")
        sensor_cfg = SimpleNamespace(name="contact")
        for _ in range(10):
            actual_asset, actual_sensor, asset_ids, sensor_ids = gait._foot_ids(env, asset_cfg, sensor_cfg)
            self.assertIs(actual_asset, asset)
            self.assertIs(actual_sensor, sensor)
            self.assertEqual(asset_ids, [3, 1, 4, 2])
            self.assertEqual(sensor_ids, [4, 3, 2, 1])
        self.assertEqual(sensor.metadata_reads, 1)

        other = Sensor([*gait.FOOT_NAMES, "base"])
        other_scene = Scene(robot=asset)
        other_scene.sensors = {"contact": other}
        other_env = SimpleNamespace(scene=other_scene)
        self.assertEqual(gait._foot_ids(other_env, asset_cfg, sensor_cfg)[3], [0, 1, 2, 3])
        self.assertEqual(other.metadata_reads, 1)
        self.assertEqual(sensor.metadata_reads, 1)

    def test_frequency_is_bounded_and_direction_independent(self):
        command = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.65, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ]
        )
        actual = mdp.gait_frequency_from_command(command, 1.5, 2.5, (0.3, 1.0))
        torch.testing.assert_close(actual, torch.tensor([1.5, 1.5, 2.0, 2.5, 2.5]))

    def test_frequency_r6_extends_only_above_one_meter_per_second(self):
        env = fake_env(torch.tensor([
            [0.3, 0.0, 0.0], [1.0, 0.0, 0.0], [1.25, 0.0, 0.0],
            [1.5, 0.0, 0.0], [2.0, 0.0, 0.0],
        ]))
        env.cfg.gait_frequency_high_speed_hz = 3.0
        env.cfg.gait_frequency_high_speed_m_s = 1.5
        actual = mdp.gait_frequency(env)
        torch.testing.assert_close(
            actual, torch.tensor([1.5, 2.5, 2.75, 3.0, 3.0])
        )

    def test_phase_and_targets_share_the_speed_dependent_clock(self):
        env = fake_env(torch.tensor([[0.65, 0.0, 0.0]]), torch.tensor([10.0]))
        phase = mdp.gait_phase(env)
        angle = 0.4 * 2.0 * math.pi
        torch.testing.assert_close(
            phase, torch.tensor([[math.sin(angle), math.cos(angle)]])
        )

        stance, clearance, stable = mdp.gait_targets(env)
        self.assertEqual(stance.tolist(), [[True, False, False, True]])
        expected_height = 0.04 * math.sin(0.75 * math.pi)
        torch.testing.assert_close(
            clearance,
            torch.tensor([[0.0, expected_height, expected_height, 0.0]]),
        )
        self.assertTrue(stable.all().item())

    def test_phase_is_zero_for_static_command_and_active_for_yaw(self):
        env = fake_env(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]),
            torch.tensor([5.0, 5.0]),
        )
        phase = mdp.gait_phase(env)
        torch.testing.assert_close(phase[0], torch.zeros(2))
        self.assertGreater(torch.linalg.vector_norm(phase[1]).item(), 0.999)

    def test_excess_clearance_ignores_stance_and_invalid_rays(self):
        clearance = torch.tensor([[0.08, float("nan"), 0.10, 0.05]])
        target = torch.tensor([[0.04, 0.04, 0.04, 0.04]])
        stance = torch.tensor([[False, False, True, False]])
        valid = torch.tensor([[True, False, True, True]])
        actual = mdp.excess_swing_clearance(clearance, target, stance, valid, 0.02)
        torch.testing.assert_close(actual, torch.tensor([0.0002]))

    def test_pitch_terms_use_gravity_x_and_body_pitch_rate(self):
        data = SimpleNamespace(
            projected_gravity_b=torch.tensor([[0.2, -0.3, -0.9], [-0.1, 0.0, -1.0]]),
            root_ang_vel_b=torch.tensor([[1.0, -0.4, 2.0], [0.0, 0.3, 0.0]]),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        cfg = SimpleNamespace(name="robot")
        torch.testing.assert_close(
            mdp.body_pitch_l2(env, cfg), torch.tensor([0.04, 0.01])
        )
        torch.testing.assert_close(
            mdp.body_pitch_rate_l2(env, cfg), torch.tensor([0.16, 0.09])
        )


if __name__ == "__main__":
    unittest.main()
