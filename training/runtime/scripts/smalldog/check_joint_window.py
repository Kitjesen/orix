"""Verify imported joint bounds, unclipped nominal actions and controlled leg motion."""
# Isaac modules must be imported after SimulationApp starts.
# ruff: noqa: E402
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("static", "swing", "expanded"), default="static")
parser.add_argument("--output")
parser.add_argument("--trajectory", default="validation/joint_window_probe_trajectory.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from robot_lab.assets.smalldog import ASSET_DIR, CONTRACT, JOINT_NAMES
from robot_lab.tasks.smalldog.env_cfg import SmallDogEnvCfg

cfg = SmallDogEnvCfg()
cfg.scene.num_envs = 8 if args.mode == "static" else 4
cfg.sim.device = args.device
cfg.seed = 42
for name in vars(cfg.events):
    if name.startswith("randomize_"):
        setattr(cfg.events, name, None)
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.events.reset_base.params["pose_range"] = {}
cfg.events.reset_base.params["velocity_range"] = {}
cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
if args.mode != "static":
    cfg.scene.robot.spawn.fix_base = True
    cfg.scene.robot.spawn.usd_file_name = CONTRACT["version"] + "_fixed_probe.usd"
    cfg.terminations.illegal_contact = None
    cfg.terminations.bad_orientation = None

env = gym.make("RobotLab-SmallDog-V67-Flat-v0", cfg=cfg)
obs, _ = env.reset()
u = env.unwrapped
robot = u.scene["robot"]
ids, joint_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
foot_names = ["fl_foot_Link", "fr_foot_Link", "rl_foot_Link", "rr_foot_Link"]
foot_ids, resolved = robot.find_bodies(foot_names, preserve_order=True)
assert resolved == foot_names
root = ET.parse(ASSET_DIR / "urdf/smalldog_v67.urdf").getroot()
source_limits = {joint.get("name"): [float(joint.find("limit").get(key)) for key in ("lower", "upper")]
                 for joint in root.findall("joint") if joint.get("type") == "revolute"}
hard = robot.data.joint_pos_limits[:, ids].clone()
wanted = torch.tensor([source_limits[name] for name in joint_names], device=u.device)
assert torch.allclose(hard, wanted.unsqueeze(0).expand_as(hard), atol=1e-6), "USD contains stale URDF limits"
soft = robot.data.soft_joint_pos_limits[:, ids].clone()
default = robot.data.default_joint_pos[:, ids].clone()
term = u.action_manager.get_term("joint_pos")
original_action = torch.zeros(u.num_envs, 12, device=u.device)
clip_tests = []
for index in range(12):
    for sign in (-1.0, 1.0):
        action = original_action.clone()
        action[:, index] = sign
        term.process_actions(action)
        error = (term.raw_actions * term._scale + term._offset - term.processed_actions).abs()
        assert float(error.max()) < 1e-6, (joint_names[index], sign, float(error.max()))
        assert ((term.processed_actions >= soft[..., 0] - 1e-6)
                & (term.processed_actions <= soft[..., 1] + 1e-6)).all()
        clip_tests.append({"joint": joint_names[index], "raw_action": sign, "max_clip_rad": float(error.max())})

resets, samples = 0, []
for step in range(250):
    obs, reward, terminated, truncated, _ = env.step(original_action)
    assert torch.isfinite(reward).all()
    assert all(torch.isfinite(value).all() for value in obs.values())
    resets += int(terminated.sum())
    if step >= 50:
        samples.append({"root_z": robot.data.root_pos_w[:, 2].clone(),
                        "upright": -robot.data.projected_gravity_b[:, 2].clone(),
                        "q": robot.data.joint_pos[:, ids].clone(),
                        "tau": robot.data.applied_torque[:, ids].clone()})
assert resets == 0, f"Zero-action stance terminated {resets} times"
assert float(torch.stack([x["upright"] for x in samples]).min()) > 0.98

swing_report = None
if args.mode != "static":
    data = json.loads(Path(args.trajectory).read_text())
    command_sequence = torch.tensor(data["joint_positions_rad"], device=u.device)
    assert data["joint_order"] == joint_names
    # Each environment probes one leg; the fixed base isolates PD movement from balance.
    prefixes = ("fl", "fr", "rl", "rr")
    measured = []
    joint_trajectory = []
    body_poses = []
    baseline = robot.data.body_link_pos_w[range(4), foot_ids].clone()
    for frame, goal in enumerate(command_sequence):
        targets = default.clone()
        for env_id, prefix in enumerate(prefixes):
            selected = [i for i, name in enumerate(joint_names) if name.startswith(prefix + "_")]
            targets[env_id, selected] = goal[selected]
        actions = (targets - term._offset) / term._scale
        unclipped = actions * term._scale + term._offset
        assert ((unclipped >= soft[..., 0] - 1e-6) & (unclipped <= soft[..., 1] + 1e-6)).all()
        for _ in range(4):
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(robot.data.joint_pos).all()
        measured.append(robot.data.body_link_pos_w[range(4), foot_ids].clone())
        joint_trajectory.append(robot.data.joint_pos[:, ids].clone())
        body_poses.append(robot.data.body_link_state_w[1, :, :7].cpu().tolist())
    points = torch.stack(measured)
    lift = points[..., 2].amax(dim=0) - baseline[:, 2]
    assert (lift > 0.025).all(), f"Controlled front/rear lift was insufficient: {lift.tolist()}"
    positions = torch.stack(joint_trajectory)
    actual_ranges = {}
    for env_id, prefix in enumerate(prefixes):
        for index, name in enumerate(joint_names):
            if name.startswith(prefix + "_"):
                values = positions[:, env_id, index]
                actual_ranges[name] = [float(values.min()), float(values.max())]
                if args.mode == "expanded" and "_thigh_" in name:
                    assert float(values.abs().min()) < 0.05, f"Thigh did not approach zero: {name}"
                if args.mode == "expanded" and "_calf_" in name:
                    assert float(values.abs().max()) > 1.7, f"Calf did not enter the expanded range: {name}"
    swing_report = {"foot_names": foot_names, "actual_lift_m": lift.tolist(),
                    "fixed_base": True, "trajectory_frames": len(command_sequence),
                    "controlled_PD_probe": True,
                    "actual_joint_ranges_rad": actual_ranges,
                    "note": "Scripted bounded targets, not a learned gait or free-standing locomotion result.",
                    "body_names": robot.body_names, "body_link_poses_env_fr": body_poses}

report = {"pass": True, "mode": args.mode, "asset_version": CONTRACT["version"],
          "num_envs": u.num_envs, "standing_seconds": 5.0, "zero_action_terminations": resets,
          "joint_names": joint_names, "hard_limits_rad": hard[0].tolist(), "soft_limits_rad": soft[0].tolist(),
          "default_joint_angles_rad": default[0].tolist(), "nominal_action_tests": clip_tests,
          "final_joint_angles_rad": dict(zip(joint_names, robot.data.joint_pos[0, ids].tolist())),
          "final_root_pose_w": robot.data.root_state_w[0, :7].tolist(),
          "root_z_m_mean_after_1s": float(torch.stack([x["root_z"] for x in samples]).mean()),
          "max_abs_static_torque_nm": float(torch.stack([x["tau"] for x in samples]).abs().max()),
          "min_upright_cosine_after_1s": float(torch.stack([x["upright"] for x in samples]).min()),
          "swing": swing_report}
output = Path(args.output or f"validation/joint_window_{args.mode}_report.json")
output.write_text(json.dumps(report, indent=2))
print("SMALLDOG_JOINT_WINDOW " + json.dumps({k: v for k, v in report.items() if k != "swing"}), flush=True)
if swing_report:
    print("SMALLDOG_CONTROLLED_LIFT " + json.dumps({k: v for k, v in swing_report.items() if k != "body_link_poses_env_fr"}), flush=True)
env.close()
app.close()
