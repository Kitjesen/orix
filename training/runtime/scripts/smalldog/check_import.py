"""Verify the imported V67 articulation, static support and finite observations."""
# Isaac modules must be imported after AppLauncher has started Kit.
# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v3 as iio
import robot_lab.tasks  # noqa: F401
import torch
from robot_lab.assets.smalldog import CONTRACT, JOINT_NAMES
from robot_lab.tasks.smalldog.env_cfg import SmallDogEnvCfg

out = Path('validation/v67_import')
out.mkdir(parents=True, exist_ok=True)
cfg = SmallDogEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
cfg.seed = 42
for name in vars(cfg.events):
    if name.startswith('randomize_'):
        setattr(cfg.events, name, None)
cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
cfg.events.reset_base.params['pose_range'] = {}
cfg.commands.base_velocity.ranges.lin_vel_x = (0, 0)
cfg.commands.base_velocity.ranges.lin_vel_y = (0, 0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0, 0)
env = gym.make('RobotLab-SmallDog-V67-Flat-v0', cfg=cfg, render_mode='rgb_array' if args.enable_cameras else None)
u = env.unwrapped
obs, _ = env.reset()
robot = u.scene['robot']
assert robot.num_joints == 12, robot.joint_names
assert set(robot.joint_names) == set(JOINT_NAMES)
foot_ids, foot_names = robot.find_bodies('.*_foot_Link')
assert len(foot_ids) == 4, foot_names
mass = float(robot.root_physx_view.get_masses()[0].sum())
assert abs(mass - CONTRACT['mass_kg']) < 0.02, (mass, CONTRACT['mass_kg'])
if args.enable_cameras:
    for _ in range(4):
        u.sim.render()
    iio.imwrite(out / 'initial.png', env.render())
samples = []
terminated_count = 0
for step in range(250):
    obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 12), device=u.device))
    for value in obs.values():
        assert torch.isfinite(value).all(), 'Non-finite observation'
    assert torch.isfinite(robot.data.root_state_w).all()
    terminated_count += int(terminated.sum())
    if step % 25 == 0:
        samples.append({'step': step, 'root_z_m': float(robot.data.root_pos_w[0, 2]),
                        'gravity_z': float(robot.data.projected_gravity_b[0, 2]),
                        'foot_z_m': robot.data.body_pos_w[0, foot_ids, 2].tolist(),
                        'max_abs_torque_nm': float(robot.data.applied_torque.abs().max())})
if args.enable_cameras:
    iio.imwrite(out / 'after_5s.png', env.render())
report = {'mass_kg': mass, 'joint_names': robot.joint_names, 'foot_names': foot_names,
          'observations': {k: list(v.shape) for k, v in obs.items()},
          'terminated_count': terminated_count, 'samples': samples,
          'final_joint_pos': dict(zip(robot.joint_names, robot.data.joint_pos[0].tolist())),
          'pass': terminated_count == 0 and float(robot.data.projected_gravity_b[0, 2]) < -0.9}
(out / 'report.json').write_text(json.dumps(report, indent=2))
print('SMALLDOG_VALIDATION ' + json.dumps(report), flush=True)
env.close()
app.close()
if not report['pass']:
    raise SystemExit('Static standing check failed; inspect contacts and gains before training.')
