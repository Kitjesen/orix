"""Record fixed student-policy examples and the exact Isaac terrain meshes."""
# ruff: noqa: E402
import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--command-vx", type=float, default=1.0)
parser.add_argument("--levels", type=int, nargs="+", default=[5, 10])
parser.add_argument("--duration-s", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import omni.usd
import robot_lab.tasks  # noqa: F401
import torch
import trimesh
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from pxr import Usd, UsdGeom
from robot_lab.assets.smalldog import CONTRACT
from robot_lab.tasks.smalldog.terrain_r6_cfg import SmallDogTerrainR6EnvCfg, SmallDogTerrainR6RunnerCfg
from robot_lab.tasks.smalldog.terrain_state import initialize_stratified_terrain
from rsl_rl.runners import OnPolicyRunnerCTS


def world_terrain_mesh(prim_path):
    stage = omni.usd.get_context().get_stage()
    parent = stage.GetPrimAtPath(prim_path)
    xforms = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange(parent):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError("Expected triangle terrain from the configured generator")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim)).T
        vertices = trimesh.transform_points(points, transform)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    if not meshes:
        raise RuntimeError(f"No actual terrain mesh below {prim_path}")
    return trimesh.util.concatenate(meshes)


def main():
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = SmallDogTerrainR6EnvCfg()
    task = "RobotLab-SmallDog-V67-RoughR6-v0"
    cfg.scene.num_envs = 220
    cfg.sim.device = args.device
    cfg.seed = 43
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.forward_speed = None
    cfg.scene.terrain.max_init_terrain_level = 10
    for name in vars(cfg.events):
        if name.startswith("randomize_"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command_vx, args.command_vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    for group in (cfg.observations.policy, cfg.observations.single_obs):
        group.enable_corruption = False

    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    u = env.unwrapped
    initialize_stratified_terrain(u, 10)
    env.reset()
    runner = OnPolicyRunnerCTS(env, SmallDogTerrainR6RunnerCfg().to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()
    obs = env.get_observations()
    robot = u.scene["robot"]
    terrain = u.scene.terrain
    generator = cfg.scene.terrain.terrain_generator
    names = list(generator.sub_terrains)
    proportions = np.array([part.proportion for part in generator.sub_terrains.values()])
    column_type = np.searchsorted(np.cumsum(proportions) / proportions.sum(),
                                  (np.arange(generator.num_cols) + 0.5) / generator.num_cols)
    selected = []
    for level in args.levels:
        if not 0 <= level < generator.num_rows:
            raise ValueError("Requested recording level is outside the grid")
        for index, name in enumerate(names):
            column = int(np.flatnonzero(column_type == index)[0])
            selected.append({"id": f"{name}_l{level:02d}", "type": name, "label": name,
                             "level": level, "column": column,
                             "selected_env_index": level * generator.num_cols + column})
    ids = torch.tensor([s["selected_env_index"] for s in selected], device=u.device)
    origins = terrain.terrain_origins.detach().cpu().numpy()
    lower_xy = origins[0, 0, :2] - np.array(generator.size) / 2
    sample_every = round(1 / (25 * u.step_dt))
    if not math.isclose(sample_every * u.step_dt, 1 / 25, abs_tol=1e-6):
        raise ValueError("Physics step does not support exact 25 fps capture")
    traces = [[] for _ in selected]
    count = len(selected)
    pending_reset = torch.zeros(count, dtype=torch.bool, device=u.device)
    reset_counts = np.zeros(count, dtype=int)
    vx_samples, pitch_samples, roll_samples, tile_samples = [], [], [], []
    all_resets = torch.zeros(220, dtype=torch.long, device=u.device)

    with torch.inference_mode():
        for step in range(round(args.duration_s / u.step_dt)):
            actions = policy(obs)
            assert torch.isfinite(actions).all()
            obs, _, dones, _ = env.step(actions)
            poses = robot.data.body_link_state_w[ids, :, :7]
            assert torch.isfinite(poses).all()
            all_resets += dones.long()
            pending_reset |= dones[ids].bool()
            reset_counts += dones[ids].cpu().numpy().astype(int)
            roots = robot.data.root_state_w[ids, :7].cpu().numpy()
            cells = np.floor((roots[:, :2] - lower_xy) / np.array(generator.size)).astype(int)
            in_tile = np.array([row == s["level"] and col == s["column"]
                                for (row, col), s in zip(cells, selected)])
            if step >= round(2 / u.step_dt):
                vx_samples.append(robot.data.root_lin_vel_b[ids, 0].cpu().numpy().copy())
                gravity = robot.data.projected_gravity_b[ids]
                pitch_samples.append(torch.asin(gravity[:, 0].clamp(-1, 1)).cpu().numpy())
                roll_samples.append(torch.atan2(-gravity[:, 1], -gravity[:, 2]).cpu().numpy())
            tile_samples.append(in_tile)
            if step % sample_every == 0:
                poses_cpu = poses.cpu().tolist()
                joints_cpu = robot.data.joint_pos[ids].cpu().tolist()
                vx_cpu = robot.data.root_lin_vel_b[ids, 0].cpu().tolist()
                for i, (row, col) in enumerate(cells):
                    inside = 0 <= row < generator.num_rows and 0 <= col < generator.num_cols
                    traces[i].append({
                        "step": step, "t": (step + 1) * u.step_dt,
                        "root_pose": roots[i].tolist(), "body_poses": poses_cpu[i],
                        "joint_pos": joints_cpu[i], "vx": vx_cpu[i],
                        "reset": bool(pending_reset[i]), "inside_initial_tile": bool(in_tile[i]),
                        "current_terrain_level": int(row) if inside else None,
                        "current_terrain_type": names[column_type[col]] if inside else "border",
                    })
                pending_reset.zero_()

    mesh = world_terrain_mesh(cfg.scene.terrain.prim_path)
    face_min = mesh.vertices[mesh.faces].min(axis=1)
    face_max = mesh.vertices[mesh.faces].max(axis=1)
    vx, pitch, roll = map(np.stack, [vx_samples, pitch_samples, roll_samples])
    inside = np.stack(tile_samples)
    manifest = {
        "checkpoint": args.checkpoint, "asset_version": CONTRACT["version"], "task": task,
        "student_policy": True, "capture_fps": 25, "duration_s": args.duration_s,
        "num_envs": 220, "command_vx_m_s": args.command_vx,
        "terrain_grid": [generator.num_rows, generator.num_cols],
        "terrain_tile_size_m": list(generator.size),
        "environment_selection": "First terrain column of each type, selected before rollout; no best-episode selection.",
        "note": "Actual Isaac body poses and the exact simulated USD terrain mesh. Initial tile labels do not imply the robot remains inside that tile. Resets and cross-tile motion are retained. Offline mesh rendering, not an RTX camera recording.",
        "scenarios": [], "all_220_env_reset_counts": all_resets.tolist(),
    }
    for i, scenario in enumerate(selected):
        trace = traces[i]
        roots = np.array([frame["root_pose"][:3] for frame in trace])
        lo, hi = roots.min(axis=0) - [2.5, 2.5, 2.0], roots.max(axis=0) + [2.5, 2.5, 2.0]
        mask = ((face_max[:, :2] >= lo[:2]) & (face_min[:, :2] <= hi[:2])).all(axis=1)
        cropped = mesh.submesh([np.flatnonzero(mask)], append=True, repair=False)
        mesh_name = scenario["id"] + "_terrain.obj"
        cropped.export(output / mesh_name)
        report = {
            "checkpoint": args.checkpoint, "asset_version": CONTRACT["version"], "task": task,
            "student_policy": True, "capture_fps": 25, "duration_s": args.duration_s,
            "command_vx_m_s": args.command_vx, "num_envs": 220,
            "body_names": robot.body_names, "joint_names": robot.joint_names,
            "selected_env_terminations": int(reset_counts[i]),
            "selected_env_mean_vx_m_s_after_2s": float(vx[:, i].mean()),
            "selected_env_mean_abs_vx_error_m_s_after_2s": float(np.abs(vx[:, i] - args.command_vx).mean()),
            "selected_env_pitch_rms_deg_after_2s": float(np.rad2deg(np.sqrt((pitch[:, i]**2).mean()))),
            "selected_env_roll_rms_deg_after_2s": float(np.rad2deg(np.sqrt((roll[:, i]**2).mean()))),
            "initial_tile_time_s": float(inside[:, i].sum() * u.step_dt),
            "trace_env0": trace, "scenario": scenario, "note": manifest["note"],
        }
        trace_name = scenario["id"] + "_trace.json"
        (output / trace_name).write_text(json.dumps(report, indent=2))
        manifest["scenarios"].append({**scenario, "trace_path": trace_name,
                                      "terrain_mesh_path": mesh_name, "terrain_mesh_frame": "world",
                                      "command_vx_m_s": args.command_vx,
                                      **{k: v for k, v in report.items() if k.startswith("selected_env_") or k == "initial_tile_time_s"}})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SMALLDOG_TERRAIN_RECORDING " + json.dumps(manifest), flush=True)
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
