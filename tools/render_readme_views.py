"""Render README images from the shipped SmallDog V67 URDF and meshes.

The images are static forward-kinematics previews of the configured standing
pose.  They are not screenshots from Isaac Sim and do not show policy output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = REPO_ROOT / "training" / "v67_cts" / "assets" / "smalldog_v67"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images"
ORTHO_SIZE = (1600, 1100)
HERO_SIZE = (1800, 1125)
BACKGROUND = "#eef2f5"
GROUND = "#e5eaee"
TEXT = "#24343f"


def vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()], dtype=float)


def transform_from_origin(node: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if node is not None:
        transform[:3, 3] = vector(node.get("xyz", "0 0 0"))
        transform[:3, :3] = Rotation.from_euler(
            "xyz", vector(node.get("rpy", "0 0 0"))
        ).as_matrix()
    return transform


def load_standing_angles(path: Path) -> dict[str, float]:
    angles: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.split(":", maxsplit=1)
        angles[name.strip()] = float(value.strip())
    return angles


def geometry_mesh(geometry: ET.Element, urdf_dir: Path) -> pv.PolyData:
    mesh_node = geometry.find("mesh")
    box_node = geometry.find("box")
    cylinder_node = geometry.find("cylinder")
    sphere_node = geometry.find("sphere")
    if mesh_node is not None:
        mesh_path = (urdf_dir / mesh_node.get("filename", "")).resolve()
        mesh = pv.read(mesh_path)
        mesh.scale(vector(mesh_node.get("scale", "1 1 1")), inplace=True)
        return mesh
    if box_node is not None:
        half = vector(box_node.get("size", "0 0 0")) / 2.0
        return pv.Box(
            bounds=(-half[0], half[0], -half[1], half[1], -half[2], half[2])
        ).triangulate()
    if cylinder_node is not None:
        return pv.Cylinder(
            direction=(0, 0, 1),
            radius=float(cylinder_node.get("radius", "0")),
            height=float(cylinder_node.get("length", "0")),
        ).triangulate()
    if sphere_node is not None:
        return pv.Sphere(radius=float(sphere_node.get("radius", "0")))
    raise ValueError(f"Unsupported geometry: {ET.tostring(geometry, encoding='unicode')}")


def calculate_link_frames(
    robot: ET.Element, angles: dict[str, float], root_height_m: float
) -> dict[str, np.ndarray]:
    base_transform = np.eye(4)
    base_transform[2, 3] = root_height_m
    frames = {"base_link": base_transform}
    pending = list(robot.findall("joint"))
    while pending:
        advanced = False
        for joint in list(pending):
            parent = joint.find("parent").get("link")
            if parent not in frames:
                continue
            motion = np.eye(4)
            if joint.get("type") in {"revolute", "continuous"}:
                name = joint.get("name", "")
                if name not in angles:
                    raise KeyError(f"Standing pose has no angle for {name}")
                axis = vector(joint.find("axis").get("xyz", "1 0 0"))
                axis /= np.linalg.norm(axis)
                motion[:3, :3] = Rotation.from_rotvec(axis * angles[name]).as_matrix()
            child = joint.find("child").get("link")
            frames[child] = (
                frames[parent]
                @ transform_from_origin(joint.find("origin"))
                @ motion
            )
            pending.remove(joint)
            advanced = True
        if not advanced:
            unresolved = [joint.get("name") for joint in pending]
            raise ValueError(f"Disconnected URDF joints: {unresolved}")
    return frames


def load_visuals(
    asset_root: Path,
) -> tuple[list[tuple[str, pv.PolyData, tuple[float, float, float]]], dict]:
    urdf_path = asset_root / "urdf" / "smalldog_v67.urdf"
    contract_path = asset_root / "config" / "model_contract.json"
    standing_path = asset_root / "config" / "standing_pose.yaml"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    angles = load_standing_angles(standing_path)
    robot = ET.parse(urdf_path).getroot()
    root_height_m = float(contract["initial_root_height_m"])
    frames = calculate_link_frames(robot, angles, root_height_m)

    visuals: list[tuple[str, pv.PolyData, tuple[float, float, float]]] = []
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for index, visual in enumerate(link.findall("visual")):
            mesh = geometry_mesh(visual.find("geometry"), urdf_path.parent)
            mesh.transform(
                frames[link_name] @ transform_from_origin(visual.find("origin")),
                inplace=True,
            )
            color_node = visual.find("material/color")
            color = (
                tuple(vector(color_node.get("rgba", "0.66 0.70 0.73 1"))[:3])
                if color_node is not None
                else (0.66, 0.70, 0.73)
            )
            visual_name = visual.get("name") or f"visual_{index}"
            visuals.append((f"{link_name}/{visual_name}", mesh, color))

    bounds = np.asarray([item[1].bounds for item in visuals], dtype=float)
    aggregate_bounds = [
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
        float(bounds[:, 4].min()),
        float(bounds[:, 5].max()),
    ]
    metadata = {
        "asset_path": "training/v67_cts/assets/smalldog_v67",
        "urdf": "training/v67_cts/assets/smalldog_v67/urdf/smalldog_v67.urdf",
        "standing_pose": (
            "training/v67_cts/assets/smalldog_v67/config/standing_pose.yaml"
        ),
        "model_contract": (
            "training/v67_cts/assets/smalldog_v67/config/model_contract.json"
        ),
        "original_source": "SmallDog training/v67_cts asset bundle",
        "model_version": contract["version"],
        "shell_source": contract["shell_source"],
        "root_height_m": root_height_m,
        "visual_count": len(visuals),
        "joint_angles_rad": angles,
        "bounds_m": aggregate_bounds,
        "ground_clearance_m": aggregate_bounds[4],
    }
    return visuals, metadata


def add_visuals(
    plotter: pv.Plotter,
    visuals: list[tuple[str, pv.PolyData, tuple[float, float, float]]],
) -> None:
    for _, mesh, color in visuals:
        plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            split_sharp_edges=True,
            ambient=0.40,
            diffuse=0.66,
            specular=0.16,
            specular_power=28,
        )


def add_ground(plotter: pv.Plotter) -> None:
    plotter.add_mesh(
        pv.Plane(center=(0, 0, -0.004), direction=(0, 0, 1), i_size=2, j_size=2),
        color=GROUND,
        lighting=False,
    )


def annotate_image(
    path: Path,
    title: str,
    subtitle: str,
    parallel_scale: float | None = None,
) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=31)
    small = ImageFont.load_default(size=22)
    draw.rounded_rectangle((30, 28, 675, 128), radius=16, fill=(238, 242, 245, 226))
    draw.text((55, 47), title, fill=TEXT, font=font)
    draw.text((56, 87), subtitle, fill="#536672", font=small)
    if parallel_scale is not None:
        pixels_per_m = image.height / (2.0 * parallel_scale)
        scale_px = round(0.1 * pixels_per_m)
        x0, y0 = 58, image.height - 72
        draw.line((x0, y0, x0 + scale_px, y0), fill="#24343f", width=5)
        draw.line((x0, y0 - 9, x0, y0 + 9), fill="#24343f", width=4)
        draw.line(
            (x0 + scale_px, y0 - 9, x0 + scale_px, y0 + 9),
            fill="#24343f",
            width=4,
        )
        draw.text((x0, y0 - 39), "100 mm", fill=TEXT, font=small)
    image.save(path, optimize=True)


def render_view(
    visuals: list[tuple[str, pv.PolyData, tuple[float, float, float]]],
    output_path: Path,
    camera_position: tuple[float, float, float],
    focal_point: tuple[float, float, float],
    view_up: tuple[float, float, float],
    parallel_scale: float,
    title: str,
    subtitle: str,
    ground: bool,
    window_size: tuple[int, int],
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing("ssaa")
    add_visuals(plotter, visuals)
    if ground:
        add_ground(plotter)
    plotter.enable_parallel_projection()
    plotter.camera_position = [camera_position, focal_point, view_up]
    plotter.camera.parallel_scale = parallel_scale
    plotter.reset_camera_clipping_range()
    plotter.screenshot(str(output_path), transparent_background=False)
    plotter.close()
    annotate_image(output_path, title, subtitle, parallel_scale)


def compose_three_views(output_dir: Path) -> Path:
    names = ["front.png", "side.png", "top.png"]
    images = [Image.open(output_dir / name).convert("RGB") for name in names]
    panel_width = 1000
    panel_height = round(images[0].height * panel_width / images[0].width)
    resized = [image.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for image in images]
    gap = 18
    header = 118
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gap * 4, panel_height + header + gap),
        color=BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=38)
    note_font = ImageFont.load_default(size=22)
    draw.text((gap, 22), "ORIX / SmallDog V67 - orthographic views", fill=TEXT, font=title_font)
    draw.text(
        (gap, 72),
        "Identical scale in all panels | standing-pose FK | robot front is +X",
        fill="#536672",
        font=note_font,
    )
    for index, image in enumerate(resized):
        x = gap + index * (panel_width + gap)
        canvas.paste(image, (x, header))
    result = output_dir / "three_views.png"
    canvas.save(result, optimize=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals, metadata = load_visuals(args.asset_root.resolve())
    xmin, xmax, ymin, ymax, zmin, zmax = metadata["bounds_m"]
    center = np.asarray(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    ortho_scale = span * 0.64

    render_view(
        visuals,
        args.output_dir / "appearance.png",
        tuple(center + np.asarray((1.10, -1.35, 0.80))),
        tuple(center),
        (0, 0, 1),
        span * 0.61,
        "ORIX / SmallDog V67",
        "3/4 appearance | configured standing pose",
        True,
        HERO_SIZE,
    )
    render_view(
        visuals,
        args.output_dir / "front.png",
        tuple(center + np.asarray((2.2, 0, 0))),
        tuple(center),
        (0, 0, 1),
        ortho_scale,
        "FRONT",
        "camera at +X | looking toward origin",
        True,
        ORTHO_SIZE,
    )
    render_view(
        visuals,
        args.output_dir / "side.png",
        tuple(center + np.asarray((0, -2.2, 0))),
        tuple(center),
        (0, 0, 1),
        ortho_scale,
        "SIDE",
        "camera at -Y | robot front points left-to-right (+X)",
        True,
        ORTHO_SIZE,
    )
    render_view(
        visuals,
        args.output_dir / "top.png",
        tuple(center + np.asarray((0, 0, 2.2))),
        tuple(center),
        (1, 0, 0),
        ortho_scale,
        "TOP",
        "+X robot front is upward in this image",
        False,
        ORTHO_SIZE,
    )
    compose_three_views(args.output_dir)
    metadata.update(
        {
            "render_kind": "URDF static standing-pose forward kinematics",
            "simulation": False,
            "policy_output": False,
            "projection": "orthographic",
            "orthographic_parallel_scale_m": ortho_scale,
            "orthographic_scale_identical": True,
            "views": {
                "appearance.png": "3/4 camera",
                "front.png": "camera +X looking toward origin",
                "side.png": "camera -Y looking toward origin",
                "top.png": "camera +Z; +X points up in image",
                "three_views.png": "front, side, and top composite",
            },
        }
    )
    metadata_path = args.output_dir / "render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
