"""Check contact geometry, phase reward contracts and an actual Isaac rollout."""
# Isaac modules must be imported after SimulationApp starts.
# ruff: noqa: E402
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rough", action="store_true")
parser.add_argument("--output")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.managers import SceneEntityCfg
from robot_lab.assets.smalldog import ASSET_DIR, JOINT_NAMES
from robot_lab.tasks.smalldog import gait_mdp
from robot_lab.tasks.smalldog.gait_cfg import SmallDogFlatGaitEnvCfg, SmallDogGaitEnvCfg

urdf = ET.parse(ASSET_DIR / "urdf/smalldog_v67.urdf").getroot()
for name in gait_mdp.FOOT_NAMES:
    collision = urdf.find(f"./link[@name='{name}']/collision")
    offset = [float(x) for x in collision.find("origin").get("xyz").split()]
    assert torch.allclose(torch.tensor(offset), torch.tensor(gait_mdp.FOOT_SPHERE_OFFSET), atol=1e-8)
    assert abs(float(collision.find("geometry/sphere").get("radius")) - gait_mdp.FOOT_SPHERE_RADIUS) < 1e-8

clock = SimpleNamespace(episode_length_buf=torch.arange(1000), step_dt=0.02, device="cpu",
                        cfg=SimpleNamespace(gait_frequency_hz=1.5, gait_stance_fraction=0.65, gait_clearance_m=0.04))
expected, target, stable = gait_mdp.gait_targets(clock)
assert expected[10].tolist() == [True, False, False, True]
assert expected[27].tolist() == [False, True, True, False]
eligible = stable.all(dim=-1)
assert (target[expected].abs() < 1e-6).all()
assert 0.039 < float(target.max()) <= 0.040001
patterns = torch.tensor([[bool(mask & (1 << i)) for i in range(4)] for mask in range(16)])
for index in (10, 27):
    wanted = expected[index].expand(16, 4)
    scores = gait_mdp.phase_contact_score(patterns, wanted)
    assert (scores > 0).sum() == 1
    assert torch.equal(patterns[scores.argmax()], expected[index])
    assert (scores[patterns.sum(dim=-1) == 3] < 0).all()
fixed = torch.tensor([True, False, False, True]).expand(1000, 4)
fixed_score = float(gait_mdp.phase_contact_score(fixed, expected)[eligible].mean())
all_contact_score = float(gait_mdp.phase_contact_score(torch.ones_like(expected), expected)[eligible].mean())
assert fixed_score <= 0 and all_contact_score <= 0
assert torch.all(gait_mdp.phase_contact_score(expected, expected) == 1)
velocity = gait_mdp.contact_point_velocity(torch.tensor([[1., 0., 0.]]),
    torch.tensor([[0., 0., 2.]]), torch.tensor([[0., -0.5, 0.]]), torch.zeros(1, 3))
assert torch.allclose(velocity, torch.tensor([[2., 0., 0.]]))
assert torch.exp(-((torch.tensor(0.08) - 0.04) / 0.02).square()) < 0.02

cfg = SmallDogGaitEnvCfg() if args.rough else SmallDogFlatGaitEnvCfg()
cfg.scene.num_envs = 8
cfg.sim.device = args.device
cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
cfg.commands.base_velocity.rel_standing_envs = 0.0
task = "RobotLab-SmallDog-V67-RoughGait-v0" if args.rough else "RobotLab-SmallDog-V67-FlatGait-v0"
env = gym.make(task, cfg=cfg)
obs, _ = env.reset()
u = env.unwrapped
robot = u.scene["robot"]
joint_ids, joint_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
hard_limits = robot.data.joint_pos_limits[:, joint_ids]
soft_limits = robot.data.soft_joint_pos_limits[:, joint_ids]
midpoint = hard_limits.mean(dim=-1, keepdim=True)
assert torch.allclose(soft_limits, midpoint + 0.9 * (hard_limits - midpoint), atol=1e-6)
default_q = robot.data.default_joint_pos[:, joint_ids]
assert ((default_q >= soft_limits[..., 0]) & (default_q <= soft_limits[..., 1])).all()
assert list(obs["policy"].shape) == [8, 470]
assert list(obs["critic"].shape) == [8, 277]
assert list(obs["single_obs"].shape) == [8, 47]
terms = list(u.reward_manager._term_names)
assert all(name in terms for name in ("feet_regulation", "action_target_clip", "diagonal_contact", "swing_clearance"))
asset_cfg = SceneEntityCfg("robot")
sensor_cfg = SceneEntityCfg("contact_forces")
supported, measured, failures = 0, 0, 0
clearances = []
for step in range(120):
    obs, reward, terminated, truncated, _ = env.step(torch.zeros((8, 12), device=u.device))
    assert torch.isfinite(reward).all()
    assert all(torch.isfinite(value).all() for value in obs.values())
    contacts, valid, velocity = gait_mdp.foot_contact_data(u, asset_cfg, sensor_cfg)
    assert torch.isfinite(velocity).all()
    clearance, ray_valid = gait_mdp.foot_clearance(u, asset_cfg)
    assert ray_valid.all(), "A foot terrain ray missed the configured terrain"
    failures += int(terminated.sum())
    if step >= 40:
        supported += int(contacts.sum())
        measured += int((contacts & valid).sum())
        clearances.append(float(clearance.abs().mean()))
coverage = measured / max(supported, 1)
if coverage <= 0.99:
    from pxr import UsdPhysics
    print("GROUND_COLLIDERS", [str(prim.GetPath()) for prim in u.sim.stage.Traverse()
          if str(prim.GetPath()).startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI)], flush=True)
    for name in gait_mdp.FOOT_NAMES:
        sensor = u.scene.sensors[f"gait_contact_{name.split('_')[0]}"]
        print("CONTACT_DIAGNOSTIC", name, sensor.contact_physx_view.filter_count,
              sensor.data.net_forces_w[0].tolist(), sensor.data.contact_pos_w[0].tolist(), flush=True)
assert supported > 0 and coverage > 0.99, f"Ground contact point coverage is {coverage}"
u.command_manager.get_command("base_velocity").zero_()
assert (gait_mdp.gait_phase(u) == 0).all()
assert (gait_mdp.diagonal_contact_pattern(u, "base_velocity", asset_cfg, sensor_cfg) == 0).all()
assert (gait_mdp.swing_clearance_reward(u, "base_velocity", asset_cfg) == 0).all()
obs, _ = env.reset()
u.command_manager.get_command("base_velocity")[:, 0] = 0.2
phase = gait_mdp.gait_phase(u)
assert torch.allclose(phase, torch.tensor([0., 1.], device=u.device).expand_as(phase))
fresh = u.observation_manager.compute(update_history=True)
assert torch.allclose(fresh["single_obs"][:, -2:], phase)
# The observation manager must clear old episode history rather than retain its clock.
phase_history = fresh["policy"][:, -20:].reshape(8, 10, 2)
assert torch.allclose(phase_history[:, -1], phase)
assert torch.allclose(phase_history, phase.unsqueeze(1).expand_as(phase_history))

report = {"pass": True, "task": task, "num_envs": 8, "steps": 120,
          "joint_names": joint_names, "hard_limits_rad": hard_limits[0].tolist(),
          "soft_limits_rad": soft_limits[0].tolist(), "soft_limit_factor": 0.9,
          "observations": {k: list(v.shape) for k, v in obs.items()},
          "reward_terms": terms, "ground_contact_point_coverage": coverage,
          "zero_action_terminations": failures, "mean_abs_vertical_clearance_m": sum(clearances)/len(clearances),
          "fixed_diagonal_cycle_score": fixed_score, "all_contact_cycle_score": all_contact_score,
          "checks": ["URDF foot geometry", "16 contact masks", "fixed pair full-cycle",
                     "angular contact velocity", "standing gate", "phase reset and history", "finite Isaac rollout"],
          "note": "Configuration and sensor verification only; no learned gait claim."}
output = Path(args.output or ("validation/v67_gait_import_rough.json" if args.rough else "validation/v67_gait_import.json"))
output.write_text(json.dumps(report, indent=2))
print("SMALLDOG_GAIT_IMPORT " + json.dumps(report), flush=True)
env.close()
app.close()
