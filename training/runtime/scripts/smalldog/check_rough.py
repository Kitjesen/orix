"""Check generated terrain, full-episode resets, and speed-curriculum boundaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.tasks.smalldog.rough_cfg import (
    ForwardSpeedCurriculum,
    SmallDogRoughEnvCfg,
    SmallDogRoughRunnerCfg,
)
from rsl_rl.runners import OnPolicyRunnerCTS

# Synthetic outcomes verify gating; they are not policy-performance evidence.
command = SimpleNamespace(
    cfg=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(-0.15, 0.35))),
    command=torch.tensor([[0.35, 0.0, 0.0]] * 128, device=args.device),
    metrics={'error_vel_xy': torch.ones(128, device=args.device)})
fake = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command), device=args.device,
    common_step_counter=0, max_episode_length=750,
    episode_length_buf=torch.full((128,), 750, device=args.device),
    reset_time_outs=torch.ones(128, dtype=torch.bool, device=args.device))
term = object.__new__(ForwardSpeedCurriculum)
term.count = term.passed = term.completed = 0
term.error_sum = 0.0
ids = torch.arange(128, device=args.device)
assert term(fake, ids) == 0.35
assert term.count == 0
fake.common_step_counter = 750
assert term(fake, ids) == 0.35
command.metrics['error_vel_xy'].zero_()
assert term(fake, ids) == 0.45
for _ in range(10):
    command.command[:, 0] = command.cfg.ranges.lin_vel_x[1]
    term(fake, ids)
assert command.cfg.ranges.lin_vel_x[1] == 1.0
term.count = term.passed = term.completed = 0
term.error_sum = 0.0
command.cfg.ranges.lin_vel_x = (-0.15, 0.35)
command.command[:, 0] = 0.35
fake.reset_time_outs.zero_()
assert term(fake, ids) == 0.35

cfg = SmallDogRoughEnvCfg()
cfg.scene.num_envs = 80
cfg.sim.device = args.device
cfg.seed = 43
env = RslRlVecEnvWrapper(gym.make('RobotLab-SmallDog-V67-Rough-v0', cfg=cfg))
u = env.unwrapped
runner = OnPolicyRunnerCTS(env, SmallDogRoughRunnerCfg().to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint, load_optimizer=False)
policy = runner.get_inference_policy()
obs = env.get_observations()
terrain = u.scene.terrain
assert terrain.terrain_origins.shape[:2] == (8, 20)
assert int(terrain.terrain_levels.max()) == 0
initial_origins = terrain.env_origins.clone()
assert torch.isfinite(initial_origins).all()
terminations = 0
with torch.inference_mode():
    for step in range(800):
        actions = policy(obs)
        assert torch.isfinite(actions).all()
        obs, rewards, dones, extras = env.step(actions)
        terminations += int(dones.sum())
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.isfinite(u.scene['robot'].data.root_state_w).all()
        assert torch.isfinite(u.scene['height_scanner'].data.ray_hits_w).all()
        assert torch.isfinite(u.scene['height_scanner_small'].data.ray_hits_w).all()
report = {
    'pass': True, 'task': 'RobotLab-SmallDog-V67-Rough-v0', 'num_envs': 80,
    'steps': 800, 'duration_s': 16, 'terminations': terminations,
    'terrain_types': list(cfg.scene.terrain.terrain_generator.sub_terrains),
    'terrain_grid': list(terrain.terrain_origins.shape),
    'terrain_columns_covered': torch.unique(terrain.terrain_types).tolist(),
    'terrain_levels_final': torch.unique(terrain.terrain_levels).tolist(),
    'observations': {k: list(v.shape) for k, v in obs.items()},
    'command_vx_range': u.command_manager.get_term('base_velocity').cfg.ranges.lin_vel_x,
    'speed_gate_synthetic_checks': ['startup_ignored', 'failure_holds', 'success_advances', 'cap_1m_s', 'early_termination_holds'],
    'note': 'Integration smoke check, not rough-terrain locomotion qualification. Synthetic gate checks do not indicate learned speed.'}
Path('validation/v67_rough_import.json').write_text(json.dumps(report, indent=2))
print('SMALLDOG_ROUGH_VALIDATION ' + json.dumps(report), flush=True)
env.close()
app.close()
