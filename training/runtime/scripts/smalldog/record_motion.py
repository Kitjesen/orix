"""Measure the student's forward velocity after training, without exploration."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--command-vx', type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.tasks.smalldog.env_cfg import SmallDogEnvCfg
from robot_lab.tasks.smalldog.rsl_rl_cfg import SmallDogMoECTSRunnerCfg
from rsl_rl.runners import OnPolicyRunnerCTS

cfg = SmallDogEnvCfg()
cfg.scene.num_envs = 16
cfg.sim.device = args.device
cfg.seed = 43
for name in vars(cfg.events):
    if name.startswith('randomize_'):
        setattr(cfg.events, name, None)
cfg.events.reset_base.params['pose_range'] = {}
cfg.commands.base_velocity.ranges.lin_vel_x = (args.command_vx, args.command_vx)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
cfg.commands.base_velocity.rel_standing_envs = 0.0
for group in (cfg.observations.policy, cfg.observations.single_obs):
    group.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make('RobotLab-SmallDog-V67-Flat-v0', cfg=cfg))
runner_cfg = SmallDogMoECTSRunnerCfg().to_dict()
runner = OnPolicyRunnerCTS(env, runner_cfg, log_dir=None, device=args.device)
runner.load(args.checkpoint, load_optimizer=False)
policy = runner.get_inference_policy()
obs = env.get_observations()
velocities = []
terminations = 0
upright = []
trace = []
with torch.inference_mode():
    for step in range(500):
        actions = policy(obs)
        assert torch.isfinite(actions).all()
        obs, reward, dones, extras = env.step(actions)
        robot = env.unwrapped.scene['robot']
        terminations += int(dones.sum())
        if step >= 100:
            velocities.append(robot.data.root_lin_vel_b[:, 0].clone())
            upright.append((-robot.data.projected_gravity_b[:, 2]).clone())
        if step % 2 == 0:
            trace.append({'step': step, 'root_pose': robot.data.root_state_w[0, :7].tolist(),
                          'joint_pos': robot.data.joint_pos[0].tolist(),
                          'body_poses': robot.data.body_link_state_w[0, :, :7].tolist(),
                          'vx': float(robot.data.root_lin_vel_b[0, 0]),
                          'reset': bool(dones[0])})
v = torch.stack(velocities)
report = {'checkpoint': args.checkpoint, 'student_policy': True, 'command_vx_m_s': args.command_vx,
          'num_envs': 16, 'duration_s': 10, 'terminations': terminations,
          'mean_vx_m_s_after_2s': float(v.mean()),
          'mean_abs_vx_error_m_s_after_2s': float((v - args.command_vx).abs().mean()),
          'mean_upright_cosine': float(torch.stack(upright).mean()),
          'joint_names': robot.joint_names, 'body_names': robot.body_names,
          'capture_fps': 25, 'trace_env0': trace,
          'note': 'One constant flat-ground command; not a hardware or rough-terrain evaluation.'}
Path(args.output).write_text(json.dumps(report, indent=2))
print('SMALLDOG_RECORDING ' + json.dumps({k: v for k, v in report.items() if k != 'trace_env0'}), flush=True)
env.close()
app.close()
