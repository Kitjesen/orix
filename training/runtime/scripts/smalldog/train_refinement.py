"""Initialize motion refinement from policy weights with fresh optimizers."""
# ruff: noqa: E402
import argparse
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--iterations", type=int, default=400)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--rough", action="store_true")
parser.add_argument("--run-name", default="v67_r4_pitch_refinement_flat_pilot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.assets.smalldog import CONTRACT
from robot_lab.tasks.smalldog.refinement_cfg import (
    SmallDogRefinedFlatGaitEnvCfg,
    SmallDogRefinedGaitEnvCfg,
    SmallDogRefinedGaitRunnerCfg,
)
from rsl_rl.runners import OnPolicyRunnerCTS

cfg = SmallDogRefinedGaitEnvCfg() if args.rough else SmallDogRefinedFlatGaitEnvCfg()
agent_cfg = SmallDogRefinedGaitRunnerCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = agent_cfg.seed
agent_cfg.device = args.device
agent_cfg.max_iterations = args.iterations
task = ("RobotLab-SmallDog-V67-RoughRefinedGait-v0" if args.rough
        else "RobotLab-SmallDog-V67-FlatRefinedGait-v0")
log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name
)
log_dir.mkdir(parents=True)
cfg.log_dir = str(log_dir.resolve())
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
try:
    runner = OnPolicyRunnerCTS(
        env, agent_cfg.to_dict(), log_dir=str(log_dir.resolve()), device=args.device
    )
    runner.load(args.checkpoint, load_optimizer=False)
    starting_iteration = runner.current_learning_iteration
    dump_yaml(str(log_dir / "params/env.yaml"), cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
    manifest = {
        "task": task, "asset_version": CONTRACT["version"],
        "checkpoint": str(Path(args.checkpoint).resolve()), "load_optimizer": False,
        "source_checkpoint_task": "RobotLab-SmallDog-V67-RoughGait-v0",
        "initialization_note": "R3 rough-policy weights initialize the R4 refinement; optimizer state is reset.",
        "starting_iteration": starting_iteration, "additional_iterations": args.iterations,
        "expected_last_iteration": starting_iteration + args.iterations - 1,
        "num_envs": args.num_envs, "log_dir": str(log_dir.resolve()),
        "gait_frequency_hz": [cfg.gait_frequency_hz, cfg.gait_frequency_max_hz],
        "gait_frequency_speed_range_m_s": list(cfg.gait_frequency_speed_range),
        "gait_stance_fraction": cfg.gait_stance_fraction,
        "gait_clearance_m": cfg.gait_clearance_m,
        "lin_vel_x_range_m_s": list(cfg.commands.base_velocity.ranges.lin_vel_x),
    }
    (log_dir / "refinement_run.json").write_text(json.dumps(manifest, indent=2))
    print("SMALLDOG_REFINEMENT_START " + json.dumps(manifest), flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    print("SMALLDOG_REFINEMENT_COMPLETE " + json.dumps({
        "final_iteration": runner.current_learning_iteration,
        "log_dir": str(log_dir.resolve()),
    }), flush=True)
finally:
    env.close()
    app.close()
