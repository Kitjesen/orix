"""Record SmallDog terrain rollouts directly from the Isaac Sim RTX viewport.

The pixels in the generated videos come from ``env.render()`` with
``render_mode='rgb_array'``.  This script does not replay poses with PyVista or
replace the simulated terrain.  It runs the loaded student policy in the real
R6 task, follows the selected robot with the Isaac viewport camera, and keeps
all reset and failure frames.
"""

# Isaac modules must be imported after AppLauncher starts Kit.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--command-vx", type=float, default=1.0)
parser.add_argument("--levels", type=int, nargs="+", default=[5, 10])
parser.add_argument("--terrain-types", nargs="+", default=None)
parser.add_argument("--duration-s", type=float, default=10.0)
parser.add_argument("--capture-fps", type=int, default=25)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--camera-eye-offset",
    type=float,
    nargs=3,
    default=(1.50, -1.80, 0.82),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--camera-target-offset",
    type=float,
    nargs=3,
    default=(0.10, 0.0, 0.08),
    metavar=("X", "Y", "Z"),
)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--smoke-frames", type=int, default=8)
parser.add_argument("--no-compilation", action="store_true")
parser.add_argument(
    "--render-call",
    choices=("sim-then-env", "env-only"),
    default="sim-then-env",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("Isaac camera recording requires --enable_cameras")
if args.capture_fps <= 0:
    parser.error("--capture-fps must be positive")
if args.width <= 0 or args.height <= 0:
    parser.error("--width and --height must be positive")
if args.smoke_frames <= 0:
    parser.error("--smoke-frames must be positive")

app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_lab.assets.smalldog import CONTRACT
from robot_lab.tasks.smalldog.terrain_r6_cfg import (
    SmallDogTerrainR6EnvCfg,
    SmallDogTerrainR6RunnerCfg,
)
from rsl_rl.runners import OnPolicyRunnerCTS


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError(f"Terrain name has no usable filename characters: {value!r}")
    return cleaned


def load_font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def root_pitch_deg(root_pose) -> float:
    qw, qx, qy, qz = [float(value) for value in root_pose[3:7]]
    sine = 2.0 * (qw * qy - qz * qx)
    return math.degrees(math.asin(max(-1.0, min(1.0, sine))))


def rgb_frame(value) -> np.ndarray:
    if value is None:
        raise RuntimeError("Isaac env.render() returned None")
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"Expected HxWx3/4 RGB array, got shape {frame.shape}")
    frame = frame[:, :, :3]
    if np.issubdtype(frame.dtype, np.floating):
        if not np.isfinite(frame).all():
            raise RuntimeError("Isaac RGB frame contains non-finite pixels")
        scale = 255.0 if float(frame.max(initial=0.0)) <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
    if float(frame.std()) < 1.0:
        raise RuntimeError(
            "Isaac RGB frame is nearly uniform; inspect RTX/Vulkan/viewport setup"
        )
    return np.ascontiguousarray(frame)


def annotate_frame(
    frame: np.ndarray,
    *,
    checkpoint: str,
    start_type: str,
    start_level: int,
    current_terrain: str,
    time_s: float,
    vx: float,
    pitch_deg: float,
    reset: bool,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = load_font(max(22, image.width // 48))
    body_font = load_font(max(16, image.width // 70))
    draw.rectangle((0, 0, image.width, 50), fill=(7, 18, 27, 182))
    draw.rectangle((0, image.height - 64, image.width, image.height), fill=(7, 18, 27, 182))
    draw.text(
        (16, 10),
        f"SmallDog V67 | START {start_type} | Start level {start_level} | Student policy",
        font=title_font,
        fill=(240, 248, 252, 255),
    )
    reset_text = " | RESET EVENT" if reset else ""
    draw.text(
        (16, image.height - 52),
        f"{checkpoint} | t {time_s:5.2f} s | vx {vx:+.3f} m/s | pitch {pitch_deg:+.1f} deg{reset_text}",
        font=body_font,
        fill=(232, 244, 250, 255),
    )
    draw.text(
        (16, image.height - 28),
        f"Current terrain: {current_terrain} | Direct Isaac Sim RTX RGB",
        font=body_font,
        fill=(126, 225, 255, 255),
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def terrain_layout(generator):
    names = list(generator.sub_terrains)
    proportions = np.asarray(
        [part.proportion for part in generator.sub_terrains.values()], dtype=float
    )
    column_types = np.searchsorted(
        np.cumsum(proportions) / proportions.sum(),
        (np.arange(generator.num_cols) + 0.5) / generator.num_cols,
    )
    first_columns = {
        name: int(np.flatnonzero(column_types == index)[0])
        for index, name in enumerate(names)
    }
    return names, column_types, first_columns


def current_terrain_label(
    root_xy: np.ndarray,
    lower_xy: np.ndarray,
    tile_size_xy: np.ndarray,
    rows: int,
    columns: int,
    names: list[str],
    column_types: np.ndarray,
) -> tuple[str, int | None, str]:
    row, column = np.floor((root_xy - lower_xy) / tile_size_xy).astype(int)
    if not (0 <= row < rows and 0 <= column < columns):
        return "border", None, "BORDER / outside terrain grid"
    terrain_type = names[int(column_types[column])]
    return terrain_type, int(row), f"{terrain_type} | level {int(row)}"


def camera_view(sim, root_position: np.ndarray) -> None:
    eye = root_position + np.asarray(args.camera_eye_offset, dtype=float)
    target = root_position + np.asarray(args.camera_target_offset, dtype=float)
    sim.set_camera_view(eye.tolist(), target.tolist())


def assign_scenario(u, level: int, column: int) -> None:
    terrain = u.scene.terrain
    terrain.terrain_levels.fill_(level)
    terrain.terrain_types.fill_(column)
    terrain.env_origins[:] = terrain.terrain_origins[level, column]
    u.episode_length_buf.zero_()
    u.common_step_counter = 0


def main() -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).name

    cfg = SmallDogTerrainR6EnvCfg()
    task = "RobotLab-SmallDog-V67-RoughR6-v0"
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    control_dt = cfg.sim.dt * cfg.decimation
    configured_capture_stride = round(1.0 / (args.capture_fps * control_dt))
    cfg.sim.render_interval = configured_capture_stride * cfg.decimation
    cfg.viewer.resolution = (args.width, args.height)
    cfg.seed = 43
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.forward_speed = None
    cfg.scene.terrain.max_init_terrain_level = 10
    cfg.scene.sky_light.spawn.intensity = 2400.0
    cfg.scene.sky_light.spawn.color = (0.95, 0.97, 1.0)
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

    gym_env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    u = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)
    runner = OnPolicyRunnerCTS(
        env, SmallDogTerrainR6RunnerCfg().to_dict(), log_dir=None, device=args.device
    )
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()
    robot = u.scene["robot"]
    terrain = u.scene.terrain
    generator = cfg.scene.terrain.terrain_generator
    names, column_types, first_columns = terrain_layout(generator)

    requested_types = args.terrain_types or names
    unknown_types = sorted(set(requested_types) - set(names))
    if unknown_types:
        raise ValueError(f"Unknown terrain types: {unknown_types}; available={names}")
    requested_levels = list(args.levels)
    for level in requested_levels:
        if not 0 <= level < generator.num_rows:
            raise ValueError(
                f"Terrain level {level} is outside 0..{generator.num_rows - 1}"
            )
    if args.smoke:
        requested_types = ["obstacles" if "obstacles" in names else names[0]]
        requested_levels = [min(5, generator.num_rows - 1)]

    capture_stride = round(1.0 / (args.capture_fps * u.step_dt))
    if not math.isclose(
        capture_stride * u.step_dt, 1.0 / args.capture_fps, abs_tol=1e-6
    ):
        raise ValueError(
            f"Control dt {u.step_dt} cannot produce exact {args.capture_fps} fps"
        )
    if args.smoke:
        frame_count = args.smoke_frames
        total_steps = frame_count * capture_stride
    else:
        total_steps = round(args.duration_s / u.step_dt)
        frame_count = total_steps // capture_stride
        expected_frames = round(args.duration_s * args.capture_fps)
        if frame_count != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} frames, computed {frame_count}"
            )

    lower_xy = terrain.terrain_origins[0, 0, :2].detach().cpu().numpy()
    lower_xy -= np.asarray(generator.size, dtype=float) / 2.0
    tile_size_xy = np.asarray(generator.size, dtype=float)
    scenarios = [
        {
            "id": f"{safe_name(terrain_type)}_l{level:02d}",
            "type": terrain_type,
            "level": level,
            "column": first_columns[terrain_type],
        }
        for level in requested_levels
        for terrain_type in requested_types
    ]

    compilation_writer = None
    compilation_path = output_dir / "smalldog_isaac_terrain_compilation.mp4"
    if not args.no_compilation and not args.smoke:
        compilation_writer = imageio.get_writer(
            compilation_path,
            fps=args.capture_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

    reports = []
    try:
        for scenario_index, scenario in enumerate(scenarios, start=1):
            scenario_id = scenario["id"]
            scenario_dir = output_dir / scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)
            assign_scenario(u, scenario["level"], scenario["column"])
            with torch.inference_mode():
                gym_env.reset()
                obs = env.get_observations()
            reset_count = 0
            pending_reset = False
            traces = []

            root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
            camera_view(u.sim, root_position)
            for _ in range(8):
                u.sim.render()
            warmup = rgb_frame(gym_env.render())
            if warmup.shape[:2] != (args.height, args.width):
                raise RuntimeError(
                    f"Isaac render resolution is {warmup.shape[:2]}, expected "
                    f"{(args.height, args.width)}"
                )

            video_path = scenario_dir / f"{scenario_id}_isaac_rtx.mp4"
            writer = imageio.get_writer(
                video_path,
                fps=args.capture_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            first_path = scenario_dir / f"{scenario_id}_first.png"
            middle_path = scenario_dir / f"{scenario_id}_middle.png"
            last_path = scenario_dir / f"{scenario_id}_last.png"
            poster_path = scenario_dir / f"{scenario_id}_poster.png"

            captured = 0
            with torch.inference_mode():
                for step in range(total_steps):
                    actions = policy(obs)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Student policy produced non-finite actions")
                    obs, _, dones, _ = env.step(actions)
                    pending_reset |= bool(dones[0])
                    reset_count += int(dones[0])
                    if step % capture_stride:
                        continue

                    root_pose = robot.data.root_state_w[0, :7].detach().cpu().tolist()
                    root_position = np.asarray(root_pose[:3], dtype=float)
                    current_type, current_level, current_label = current_terrain_label(
                        root_position[:2],
                        lower_xy,
                        tile_size_xy,
                        generator.num_rows,
                        generator.num_cols,
                        names,
                        column_types,
                    )
                    camera_view(u.sim, root_position)
                    if args.render_call == "sim-then-env":
                        u.sim.render()
                    raw = rgb_frame(gym_env.render())
                    vx = float(robot.data.root_lin_vel_b[0, 0])
                    pitch = root_pitch_deg(root_pose)
                    time_s = (step + 1) * u.step_dt
                    annotated = annotate_frame(
                        raw,
                        checkpoint=checkpoint,
                        start_type=scenario["type"],
                        start_level=scenario["level"],
                        current_terrain=current_label,
                        time_s=time_s,
                        vx=vx,
                        pitch_deg=pitch,
                        reset=pending_reset,
                    )
                    writer.append_data(annotated)
                    if compilation_writer is not None:
                        compilation_writer.append_data(annotated)
                    if captured == 0:
                        imageio.imwrite(first_path, annotated)
                    if captured == frame_count // 2:
                        imageio.imwrite(middle_path, annotated)
                    if captured == min(37, frame_count - 1):
                        imageio.imwrite(poster_path, annotated)
                    if captured == frame_count - 1:
                        imageio.imwrite(last_path, annotated)
                    if args.smoke or pending_reset:
                        imageio.imwrite(
                            scenario_dir / f"{scenario_id}_frame{captured:03d}.png",
                            annotated,
                        )
                    traces.append(
                        {
                            "frame": captured,
                            "step": step,
                            "t": time_s,
                            "vx": vx,
                            "root_pitch_deg": pitch,
                            "reset": pending_reset,
                            "root_pose": root_pose,
                            "current_terrain_type": current_type,
                            "current_terrain_level": current_level,
                        }
                    )
                    pending_reset = False
                    captured += 1
            writer.close()
            if captured != frame_count:
                raise RuntimeError(
                    f"Scenario {scenario_id} captured {captured} frames, expected {frame_count}"
                )

            vx_values = np.asarray([item["vx"] for item in traces], dtype=float)
            pitch_values = np.asarray(
                [item["root_pitch_deg"] for item in traces], dtype=float
            )
            report = {
                **scenario,
                "scenario_index": scenario_index,
                "scenario_count": len(scenarios),
                "checkpoint": args.checkpoint,
                "asset_version": CONTRACT["version"],
                "task": task,
                "student_policy": True,
                "capture_method": "Isaac Sim RTX env.render rgb_array",
                "video": str(video_path.resolve()),
                "poster": str(poster_path.resolve()),
                "first_frame": str(first_path.resolve()),
                "middle_frame": str(middle_path.resolve()),
                "last_frame": str(last_path.resolve()),
                "resolution": [args.width, args.height],
                "capture_fps": args.capture_fps,
                "frame_count": captured,
                "duration_s": captured / args.capture_fps,
                "command_vx_m_s": args.command_vx,
                "reset_count": reset_count,
                "measured_vx_m_s": {
                    "mean": float(vx_values.mean()),
                    "minimum": float(vx_values.min()),
                    "maximum": float(vx_values.max()),
                },
                "root_pitch_deg": {
                    "mean": float(pitch_values.mean()),
                    "minimum": float(pitch_values.min()),
                    "maximum": float(pitch_values.max()),
                },
                "frames": traces,
            }
            report_path = scenario_dir / f"{scenario_id}_recording.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            reports.append({key: value for key, value in report.items() if key != "frames"})
            print(
                f"ISAAC_CAMERA_SCENARIO {scenario_index}/{len(scenarios)} "
                f"{scenario_id} frames={captured} resets={reset_count} video={video_path}",
                flush=True,
            )
    finally:
        if compilation_writer is not None:
            compilation_writer.close()

    manifest = {
        "checkpoint": args.checkpoint,
        "asset_version": CONTRACT["version"],
        "task": task,
        "capture_method": "Isaac Sim RTX env.render rgb_array",
        "smoke": args.smoke,
        "camera_eye_offset": list(args.camera_eye_offset),
        "camera_target_offset": list(args.camera_target_offset),
        "render_call": args.render_call,
        "sim_render_interval": cfg.sim.render_interval,
        "resolution": [args.width, args.height],
        "capture_fps": args.capture_fps,
        "compilation": (
            str(compilation_path.resolve()) if compilation_writer is not None else None
        ),
        "scenarios": reports,
    }
    manifest_path = output_dir / "isaac_camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "SMALLDOG_ISAAC_CAMERA "
        + json.dumps(
            {
                "manifest": str(manifest_path),
                "scenario_count": len(reports),
                "capture_method": manifest["capture_method"],
                "compilation": manifest["compilation"],
            }
        ),
        flush=True,
    )
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
