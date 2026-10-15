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
joint_ids, joint_names = env.unwrapped.scene['robot'].find_joints(
    env.unwrapped.cfg.actions.joint_pos.joint_names, preserve_order=True)
foot_ids, foot_names = env.unwrapped.scene['robot'].find_bodies('.*_foot_Link')
sensor = env.unwrapped.scene['contact_forces']
sensor_foot_ids, sensor_foot_names = sensor.find_bodies(foot_names, preserve_order=True)
assert sensor_foot_names == foot_names
action_term = env.unwrapped.action_manager.get_term('joint_pos')
samples = []
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
        if step >= 100:
            sensor = env.unwrapped.scene['contact_forces']
            forces = sensor.data.net_forces_w[:, sensor_foot_ids].norm(dim=-1)
            foot_vel = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            unclipped = action_term.raw_actions * action_term._scale + action_term._offset
            clip_mask = (unclipped - action_term.processed_actions).abs() > 1e-6
            samples.append({
                'q': robot.data.joint_pos[:, joint_ids].cpu().tolist(),
                'target_clip': clip_mask.cpu().tolist(),
                'tau': robot.data.applied_torque[:, joint_ids].cpu().tolist(),
                'foot_z': robot.data.body_link_pos_w[:, foot_ids, 2].cpu().tolist(),
                'foot_contact': (forces > 1.0).cpu().tolist(),
                'foot_xy_speed': foot_vel.cpu().tolist(),
                'air_time': sensor.data.current_air_time[:, sensor_foot_ids].cpu().tolist(),
                'feet_regulation_raw': float(env.unwrapped.reward_manager.get_term_cfg('feet_regulation').func(
                    env.unwrapped, **env.unwrapped.reward_manager.get_term_cfg('feet_regulation').params).mean()),
                'feet_air_time_raw': float(env.unwrapped.reward_manager.get_term_cfg('feet_air_time').func(
                    env.unwrapped, **env.unwrapped.reward_manager.get_term_cfg('feet_air_time').params).mean()),
            })
        if step % 5 == 0:
            trace.append({'step': step, 'root_pose': robot.data.root_state_w[0, :7].tolist(),
                          'joint_pos': robot.data.joint_pos[0].tolist()})
v = torch.stack(velocities)
report = {'checkpoint': args.checkpoint, 'student_policy': True, 'command_vx_m_s': args.command_vx,
          'num_envs': 16, 'duration_s': 10, 'terminations': terminations,
          'mean_vx_m_s_after_2s': float(v.mean()),
          'mean_abs_vx_error_m_s_after_2s': float((v - args.command_vx).abs().mean()),
          'mean_upright_cosine': float(torch.stack(upright).mean()),
          'joint_names': joint_names, 'trace_joint_names': robot.joint_names, 'foot_names': foot_names, 'sensor_body_names': sensor.body_names,
          'robot_foot_ids': foot_ids, 'sensor_foot_ids': sensor_foot_ids,
          'diagnostic_samples': samples, 'trace_env0': trace,
          'height_scan_sensor_z': float(env.unwrapped.scene['height_scanner'].data.pos_w[0,2]),
          'root_z': float(robot.data.root_pos_w[0,2]),
          'critic_height_scan_min': float(obs['critic'][:, -187:].min()),
          'critic_height_scan_max': float(obs['critic'][:, -187:].max()),
          'note': 'One constant flat-ground command; not a hardware or rough-terrain evaluation.'}
Path(args.output).write_text(json.dumps(report, indent=2))
print('SMALLDOG_EVALUATION ' + json.dumps({k: v for k, v in report.items() if k not in ('trace_env0', 'diagnostic_samples')}), flush=True)
env.close()
app.close()
