"""Continue SmallDog terrain training with persistent terrain-course checkpoints."""
# ruff: noqa: E402
import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--output", default="validation/r6_preflight.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.tasks.smalldog import refinement_mdp
from robot_lab.tasks.smalldog.terrain_r6_cfg import SmallDogTerrainR6EnvCfg, SmallDogTerrainR6RunnerCfg
from robot_lab.tasks.smalldog.terrain_state import (
    capture_terrain_state,
    initialize_stratified_terrain,
    restore_terrain_state,
    terrain_state_summary,
)
from rsl_rl.runners import OnPolicyRunnerCTS


class TerrainCheckpointRunner(OnPolicyRunnerCTS):
    def save(self, path, it, last_model, infos=None):
        info = dict(infos or {})
        info["smalldog_terrain"] = capture_terrain_state(self.env.unwrapped)
        super().save(path, it, last_model, info)
        if self.logger.log_dir is not None:
            summary = {"iteration": it, **terrain_state_summary(info["smalldog_terrain"])}
            Path(self.logger.log_dir, "terrain_state_latest.json").write_text(json.dumps(summary, indent=2))

    def load(self, path, load_optimizer=True, map_location=None):
        info = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        self.terrain_restored = bool(info and "smalldog_terrain" in info)
        if self.terrain_restored:
            with torch.inference_mode():
                restore_terrain_state(self.env.unwrapped, info["smalldog_terrain"])
                self.env.reset()
        if load_optimizer:
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        return info


cfg = SmallDogTerrainR6EnvCfg()
agent_cfg = SmallDogTerrainR6RunnerCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = agent_cfg.seed
agent_cfg.device = args.device
agent_cfg.max_iterations = args.iterations
task = "RobotLab-SmallDog-V67-RoughR6-v0"
log_dir = None
if not args.check_only:
    log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_v67_r6_terrain_speed_1p5_plus10000"
    )
    log_dir.mkdir(parents=True)
    cfg.log_dir = str(log_dir.resolve())

env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
try:
    if args.check_only:
        initialize_stratified_terrain(env.unwrapped, cfg.scene.terrain.terrain_generator.num_rows - 1)
        env.reset()
    runner = TerrainCheckpointRunner(
        env, agent_cfg.to_dict(), log_dir=str(log_dir.resolve()) if log_dir else None, device=args.device
    )
    runner.load(args.checkpoint, load_optimizer=True)
    print("SMALLDOG_R6_CHECKPOINT_LOADED", runner.current_learning_iteration, flush=True)
    loaded_iteration = runner.current_learning_iteration
    assert runner.alg.optimizer.state and runner.alg.optimizer_stu_enc.state
    obs = env.get_observations()
    for name, size in (("policy", 470), ("critic", 277), ("single_obs", 47)):
        assert list(obs[name].shape) == [args.num_envs, size]
    initial = terrain_state_summary(capture_terrain_state(env.unwrapped))
    if args.check_only:
        u = env.unwrapped
        command = u.command_manager.get_command("base_velocity")
        frequencies = {}
        for speed, expected in ((0.0, 1.5), (0.3, 1.5), (1.0, 2.5), (1.25, 2.75), (1.5, 3.0)):
            command.zero_()
            command[:, 0] = speed
            frequency = refinement_mdp.gait_frequency(u)
            assert torch.allclose(frequency, torch.full_like(frequency, expected))
            frequencies[str(speed)] = float(frequency[0])
        command.zero_()
        command[:, 0] = 1.0
        policy = runner.get_inference_policy()
        terminations = 0
        with torch.inference_mode():
            for _ in range(120):
                actions = policy(obs)
                assert torch.isfinite(actions).all()
                obs, rewards, dones, _ = env.step(actions)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                terminations += int(dones.sum())
        before = capture_terrain_state(u)
        trial = Path(args.output).with_suffix(".pt")
        runner.save(str(trial), loaded_iteration, False)
        u.scene.terrain.terrain_levels.zero_()
        u.scene.terrain.terrain_types.zero_()
        u.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = (-0.15, 0.35)
        runner.load(str(trial))
        after = capture_terrain_state(u)
        for name in ("terrain_levels", "terrain_types"):
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
        assert after["forward_speed_range"] == before["forward_speed_range"]
        assert after["forward_speed_counters"] == before["forward_speed_counters"]
        report = {"pass": True, "task": task, "num_envs": args.num_envs,
                  "source_iteration": loaded_iteration, "initial_terrain": initial,
                  "frequency_by_command_m_s": frequencies, "finite_rollout_steps": 120,
                  "rollout_terminations": terminations, "optimizer_and_student_optimizer_loaded": True,
                  "terrain_checkpoint_round_trip": True,
                  "note": "Full-range terrain wiring and finite rollout; not evidence that 1.5 m/s is learned."}
        Path(args.output).write_text(json.dumps(report, indent=2))
        print("SMALLDOG_R6_PREFLIGHT " + json.dumps(report), flush=True)
    else:
        # The saved iteration already completed; label the next update consecutively.
        runner.current_learning_iteration = loaded_iteration + 1
        manifest = {"pid": os.getpid(), "task": task, "num_envs": args.num_envs,
                    "source_checkpoint": str(Path(args.checkpoint).resolve()),
                    "source_iteration": loaded_iteration, "starting_iteration": loaded_iteration + 1,
                    "additional_iterations": args.iterations,
                    "expected_last_iteration": loaded_iteration + args.iterations,
                    "load_optimizer": True, "terrain_state_restored": runner.terrain_restored,
                    "terrain_initialization": "restored" if runner.terrain_restored else "uniform_generated_levels",
                    "initial_terrain": initial, "log_dir": str(log_dir.resolve())}
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        (log_dir / "continuation.json").write_text(json.dumps(manifest, indent=2))
        Path("validation/r6_latest_run.json").write_text(json.dumps(manifest, indent=2))
        print("SMALLDOG_R6_START " + json.dumps(manifest), flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == manifest["expected_last_iteration"]
        print("SMALLDOG_R6_COMPLETE " + str(runner.current_learning_iteration), flush=True)
except Exception:
    print("SMALLDOG_R6_ERROR " + traceback.format_exc(), flush=True)
    raise
finally:
    env.close()
    app.close()
