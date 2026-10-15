"""Profile a bounded rollout with the current SmallDog training configuration."""
# ruff: noqa: E402
import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--warmup", type=int, default=12)
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
    SmallDogRefinedGaitEnvCfg,
    SmallDogRefinedGaitRunnerCfg,
)
from rsl_rl.runners import OnPolicyRunnerCTS

cfg = SmallDogRefinedGaitEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.terrain.max_init_terrain_level = 2
cfg.commands.base_velocity.ranges.lin_vel_x = (-0.15, 1.0)
cfg.sim.device = args.device
cfg.seed = 42
runner_cfg = SmallDogRefinedGaitRunnerCfg()
env = RslRlVecEnvWrapper(gym.make("RobotLab-SmallDog-V67-RoughRefinedGait-v0", cfg=cfg))
try:
    runner = OnPolicyRunnerCTS(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args.warmup):
            obs, _, _, _ = env.step(policy(obs))
        asset_cfg = SceneEntityCfg("robot")
        sensor_cfg = SceneEntityCfg("contact_forces")
        optimized = gait_mdp._foot_ids

        def original_foot_ids(environment, asset_config, sensor_config):
            asset = environment.scene[asset_config.name]
            sensor = environment.scene.sensors[sensor_config.name]
            asset_ids, names = asset.find_bodies(gait_mdp.FOOT_NAMES, preserve_order=True)
            sensor_ids = [sensor.body_names.index(name) for name in names]
            return asset, sensor, asset_ids, sensor_ids

        def contact_outputs():
            return (
                *gait_mdp.foot_contact_data(env.unwrapped, asset_cfg, sensor_cfg),
                gait_mdp.feet_contact_slip_l2(env.unwrapped, asset_cfg, sensor_cfg),
                refinement_mdp.diagonal_contact_pattern(
                    env.unwrapped, "base_velocity", asset_cfg, sensor_cfg
                ),
            )

        try:
            gait_mdp._foot_ids = original_foot_ids
            reference = contact_outputs()
        finally:
            gait_mdp._foot_ids = optimized
        actual = contact_outputs()
        for expected, value in zip(reference, actual):
            torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize()
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.enable()
        for _ in range(args.steps):
            obs, _, _, _ = env.step(policy(obs))
        torch.cuda.synchronize()
        profile.disable()
        elapsed = time.perf_counter() - start
    output = Path(args.output)
    profile.dump_stats(str(output.with_suffix(".pstats")))
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": function,
                     "primitive_calls": primitive, "calls": calls,
                     "own_s": own, "cumulative_s": cumulative})
    report = {"num_envs": args.num_envs, "steps": args.steps,
              "checkpoint": args.checkpoint, "contact_and_reward_equivalence": True,
              "max_initial_terrain_level": 2, "elapsed_s": elapsed,
              "env_steps_per_s": args.num_envs * args.steps / elapsed,
              "top_cumulative": sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[:60],
              "top_own": sorted(rows, key=lambda row: row["own_s"], reverse=True)[:60],
              "note": "Separate bounded profile, sharing the GPU with the running trainer; not a training-quality evaluation."}
    output.write_text(json.dumps(report, indent=2))
    print("SMALLDOG_SAMPLING_PROFILE " + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
