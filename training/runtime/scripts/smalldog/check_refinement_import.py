"""Check R4 phase/reward wiring, terrain contacts and checkpoint compatibility."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--rough", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.tasks.smalldog import gait_mdp, refinement_mdp
from robot_lab.tasks.smalldog.refinement_cfg import (
    SmallDogRefinedFlatGaitEnvCfg,
    SmallDogRefinedGaitEnvCfg,
    SmallDogRefinedGaitRunnerCfg,
)
from rsl_rl.runners import OnPolicyRunnerCTS

cfg = SmallDogRefinedGaitEnvCfg() if args.rough else SmallDogRefinedFlatGaitEnvCfg()
cfg.scene.num_envs = 8
cfg.sim.device = args.device
cfg.seed = 43
cfg.curriculum.terrain_levels = None
cfg.curriculum.forward_speed = None
cfg.commands.base_velocity.rel_standing_envs = 0
cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
task = ("RobotLab-SmallDog-V67-RoughRefinedGait-v0" if args.rough
        else "RobotLab-SmallDog-V67-FlatRefinedGait-v0")
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
try:
    u = env.unwrapped
    obs = env.get_observations()
    assert list(obs["policy"].shape) == [8, 470]
    assert list(obs["critic"].shape) == [8, 277]
    assert list(obs["single_obs"].shape) == [8, 47]
    agent_cfg = SmallDogRefinedGaitRunnerCfg()
    runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    assert torch.isfinite(runner.get_inference_policy()(obs)).all()
    assert u.reward_manager.get_term_cfg("diagonal_contact").func is refinement_mdp.diagonal_contact_pattern
    assert u.reward_manager.get_term_cfg("swing_clearance").func is refinement_mdp.swing_clearance_reward
    assert cfg.observations.single_obs.gait_phase.func is refinement_mdp.gait_phase
    cfg_assets = SceneEntityCfg("robot")
    cfg_contacts = SceneEntityCfg("contact_forces")
    command = u.command_manager.get_command("base_velocity")
    frequencies = {}
    for speed, expected in ((0.0, 1.5), (0.3, 1.5), (0.6, 1.9285714285714286), (1.0, 2.5)):
        command.zero_()
        command[:, 0] = speed
        frequency = refinement_mdp.gait_frequency(u, "base_velocity")
        assert torch.allclose(frequency, torch.full_like(frequency, expected))
        frequencies[str(speed)] = float(frequency[0])
        for term in (refinement_mdp.gait_phase(u),
                     refinement_mdp.diagonal_contact_pattern(u, "base_velocity", cfg_assets, cfg_contacts),
                     refinement_mdp.swing_clearance_reward(u, "base_velocity", cfg_assets),
                     refinement_mdp.excess_swing_clearance_l2(u, "base_velocity", cfg_assets)):
            assert torch.isfinite(term).all()
            if speed == 0:
                assert (term == 0).all()
    command.zero_()
    command[:, 0] = 0.3
    supported, valid_points, terminations = 0, 0, 0
    for _ in range(120):
        obs, reward, dones, _ = env.step(torch.zeros((8, 12), device=u.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        contacts, valid, velocity = gait_mdp.foot_contact_data(u, cfg_assets, cfg_contacts)
        clearance, valid_rays = gait_mdp.foot_clearance(u, cfg_assets)
        assert valid_rays.all()
        assert torch.isfinite(clearance).all() and torch.isfinite(velocity).all()
        supported += int(contacts.sum())
        valid_points += int((contacts & valid).sum())
        terminations += int(dones.sum())
    coverage = valid_points / max(supported, 1)
    assert supported > 0 and coverage > 0.99
    assert terminations == 0
    report = {
        "pass": True, "task": task, "num_envs": 8, "steps": 120,
        "observations": {key: list(value.shape) for key, value in obs.items()},
        "frequency_by_command_m_s": frequencies, "stance_fraction": cfg.gait_stance_fraction,
        "phase_reward_functions_aligned": True, "checkpoint_weights_load": True,
        "fresh_optimizer": True, "ground_contact_point_coverage": coverage,
        "zero_action_terminations": terminations,
        "reward_weights": {name: u.reward_manager.get_term_cfg(name).weight
                           for name in u.reward_manager.active_terms},
        "note": "Actual Isaac wiring/static rollout evidence; not a learned motion-quality result.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("SMALLDOG_REFINEMENT_IMPORT " + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
