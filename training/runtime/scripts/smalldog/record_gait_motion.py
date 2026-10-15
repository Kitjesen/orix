"""Record actual link poses from the final phase-observable student policy."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--command-vx", type=float, default=1.0)
parser.add_argument("--refined", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.assets.smalldog import CONTRACT
from robot_lab.tasks.smalldog.gait_cfg import SmallDogFlatGaitEnvCfg, SmallDogGaitRunnerCfg
from rsl_rl.runners import OnPolicyRunnerCTS

if args.refined:
    from robot_lab.tasks.smalldog.refinement_cfg import SmallDogRefinedFlatGaitEnvCfg, SmallDogRefinedGaitRunnerCfg

    cfg = SmallDogRefinedFlatGaitEnvCfg()
    runner_cfg = SmallDogRefinedGaitRunnerCfg()
    task = "RobotLab-SmallDog-V67-FlatRefinedGait-v0"
else:
    cfg = SmallDogFlatGaitEnvCfg()
    runner_cfg = SmallDogGaitRunnerCfg()
    task = "RobotLab-SmallDog-V67-FlatGait-v0"
cfg.scene.num_envs = 16
cfg.sim.device = args.device
cfg.seed = 43
for name in vars(cfg.events):
    if name.startswith("randomize_"):
        setattr(cfg.events, name, None)
cfg.events.reset_base.params["pose_range"] = {}
cfg.commands.base_velocity.ranges.lin_vel_x = (args.command_vx, args.command_vx)
cfg.commands.base_velocity.ranges.lin_vel_y = (0, 0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0, 0)
cfg.commands.base_velocity.rel_standing_envs = 0
for group in (cfg.observations.policy, cfg.observations.single_obs):
    group.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
runner = OnPolicyRunnerCTS(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint, load_optimizer=False)
policy = runner.get_inference_policy()
obs = env.get_observations()
u = env.unwrapped
robot = u.scene["robot"]
trace, velocities = [], []
terminations, selected_terminations = 0, 0
with torch.inference_mode():
    for step in range(500):
        actions = policy(obs)
        assert torch.isfinite(actions).all()
        obs, reward, dones, _ = env.step(actions)
        assert torch.isfinite(robot.data.body_link_state_w).all()
        terminations += int(dones.sum())
        selected_terminations += int(dones[0])
        if step >= 100:
            velocities.append(robot.data.root_lin_vel_b[:, 0].clone())
        if step % 2 == 0:
            trace.append({
                "step": step, "t": (step + 1) * u.step_dt,
                "root_pose": robot.data.root_state_w[0, :7].tolist(),
                "joint_pos": robot.data.joint_pos[0].tolist(),
                "body_poses": robot.data.body_link_state_w[0, :, :7].tolist(),
                "vx": float(robot.data.root_lin_vel_b[0, 0]), "reset": bool(dones[0]),
            })
v = torch.stack(velocities)
assert len(trace) == 250
report = {
    "asset_version": CONTRACT["version"], "checkpoint": args.checkpoint,
    "task": task, "student_policy": True, "refined": args.refined,
    "command_vx_m_s": args.command_vx, "num_envs": 16, "duration_s": 10,
    "terminations": terminations, "selected_env_terminations": selected_terminations,
    "mean_vx_m_s_after_2s": float(v.mean()),
    "selected_env_mean_vx_m_s_after_2s": float(v[:, 0].mean()),
    "mean_abs_vx_error_m_s_after_2s": float((v - args.command_vx).abs().mean()),
    "joint_names": robot.joint_names, "body_names": robot.body_names,
    "capture_fps": round(1 / (2 * u.step_dt)), "trace_env0": trace,
    "note": "Actual Isaac link poses, flat-ground deterministic student rollout. Video is reconstructed from these poses, not an RTX camera recording.",
}
Path(args.output).write_text(json.dumps(report, indent=2))
print("SMALLDOG_GAIT_RECORDING " + json.dumps({key: value for key, value in report.items() if key != "trace_env0"}), flush=True)
env.close()
app.close()
