"""Validate the portable SmallDog training asset using only the standard library."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


DEFAULT_ASSET_DIR = Path("training/v67_cts/assets/smalldog_v67")
EXPECTED_MOVING_JOINTS = 12


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def _joint_family(name: str) -> str:
    for family in ("hip", "thigh", "calf"):
        if f"_{family}_joint" in name:
            return family
    raise ValueError(f"Cannot determine joint family: {name}")


def validate(asset_dir: Path) -> dict[str, object]:
    asset_dir = asset_dir.resolve()
    urdf_path = asset_dir / "urdf" / "smalldog_v67.urdf"
    contract_path = asset_dir / "config" / "model_contract.json"
    motion_path = asset_dir.parents[1] / "joint_motion_config.json"
    errors: list[str] = []

    for required in (urdf_path, contract_path, motion_path):
        if not required.is_file():
            errors.append(f"missing required file: {required}")
    if errors:
        return {"passed": False, "asset_dir": str(asset_dir), "errors": errors}

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    root = ET.parse(urdf_path).getroot()

    mesh_refs: list[str] = []
    for mesh in root.iter("mesh"):
        reference = mesh.get("filename", "")
        mesh_refs.append(reference)
        candidate = (urdf_path.parent / reference).resolve()
        try:
            candidate.relative_to(asset_dir)
        except ValueError:
            errors.append(f"mesh reference escapes asset directory: {reference}")
            continue
        if "://" in reference or Path(reference).is_absolute():
            errors.append(f"mesh reference is not relative: {reference}")
        elif not candidate.is_file():
            errors.append(f"mesh file does not exist: {reference}")
        elif candidate.stat().st_size == 0:
            errors.append(f"mesh file is empty: {reference}")

    revolute = [joint for joint in root.findall("joint") if joint.get("type") == "revolute"]
    joint_names = [joint.get("name", "") for joint in revolute]
    if len(revolute) != EXPECTED_MOVING_JOINTS:
        errors.append(f"expected {EXPECTED_MOVING_JOINTS} revolute joints, found {len(revolute)}")
    if len(set(joint_names)) != len(joint_names):
        errors.append("duplicate revolute joint names")

    contract_order = contract.get("joint_order", [])
    if joint_names != contract_order:
        errors.append("URDF revolute joint order differs from model_contract.json joint_order")

    contract_pose = contract.get("standing_joint_angles_rad", {})
    contract_limits = contract.get("joint_position_limits_rad", {})
    motion_pose = motion.get("standing_joint_angles_rad", {})
    motion_limits = motion.get("joint_position_limits_rad", {})
    expected_set = set(joint_names)
    for label, mapping in (
        ("model contract standing pose", contract_pose),
        ("model contract limits", contract_limits),
        ("motion config standing pose", motion_pose),
        ("motion config limits", motion_limits),
    ):
        if set(mapping) != expected_set:
            errors.append(f"{label} joint names differ from URDF")

    for joint in revolute:
        name = joint.get("name", "")
        limit = joint.find("limit")
        if limit is None:
            errors.append(f"missing limit element: {name}")
            continue
        lower = float(limit.get("lower", "nan"))
        upper = float(limit.get("upper", "nan"))
        effort = float(limit.get("effort", "nan"))
        velocity = float(limit.get("velocity", "nan"))
        if not all(math.isfinite(value) for value in (lower, upper, effort, velocity)):
            errors.append(f"non-finite joint limit value: {name}")
            continue
        if not lower < upper:
            errors.append(f"invalid position limits: {name}")
        if effort <= 0 or velocity <= 0:
            errors.append(f"effort and velocity must be positive: {name}")

        if name in contract_limits:
            expected_lower, expected_upper = map(float, contract_limits[name])
            if not (_same_number(lower, expected_lower) and _same_number(upper, expected_upper)):
                errors.append(f"URDF limits differ from model contract: {name}")
        if name in motion_limits and list(contract_limits.get(name, [])) != list(motion_limits[name]):
            errors.append(f"motion config limits differ from model contract: {name}")

        pose = contract_pose.get(name)
        if pose is not None and not lower <= float(pose) <= upper:
            errors.append(f"standing pose is outside hard limits: {name}")
        if name in motion_pose and pose is not None and not _same_number(float(pose), float(motion_pose[name])):
            errors.append(f"motion config standing pose differs from model contract: {name}")

        family = _joint_family(name)
        expected_effort = float(contract["joint_effort_cap_nm"][family])
        expected_velocity = float(contract["joint_speed_cap_rad_s"][family])
        if not _same_number(effort, expected_effort):
            errors.append(f"URDF effort differs from model contract: {name}")
        if not _same_number(velocity, expected_velocity):
            errors.append(f"URDF velocity differs from model contract: {name}")

    urdf_mass = 0.0
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        if mass is None:
            errors.append(f"link has no inertial mass: {link.get('name', '')}")
            continue
        value = float(mass.get("value", "nan"))
        if not math.isfinite(value) or value <= 0:
            errors.append(f"link mass is not positive and finite: {link.get('name', '')}")
        else:
            urdf_mass += value

    base_mass_node = root.find("./link[@name='base_link']/inertial/mass")
    if base_mass_node is None:
        errors.append("base_link has no inertial mass")
    elif not _same_number(float(base_mass_node.get("value", "nan")), float(contract["base_mass_kg"])):
        errors.append("base_link mass differs from model_contract.json")
    if not _same_number(urdf_mass, float(contract["mass_kg"])):
        errors.append("total URDF mass differs from model_contract.json")

    if root.get("name") != contract.get("version"):
        errors.append("URDF robot name differs from model contract version")
    if motion.get("version") != contract.get("version"):
        errors.append("joint_motion_config.json version differs from model contract")
    for field in ("initial_root_height_m", "base_height_target_m", "soft_joint_pos_limit_factor"):
        if not _same_number(float(motion[field]), float(contract[field])):
            errors.append(f"joint motion config differs from model contract: {field}")
    if motion.get("action_scale_rad") != contract.get("action_scale_rad"):
        errors.append("joint motion action scales differ from model contract")

    return {
        "passed": not errors,
        "asset_dir": str(asset_dir),
        "version": contract.get("version"),
        "urdf": urdf_path.relative_to(asset_dir).as_posix(),
        "links": len(root.findall("link")),
        "revolute_joints": len(revolute),
        "mesh_references": len(mesh_refs),
        "unique_mesh_references": len(set(mesh_refs)),
        "mass_kg": urdf_mass,
        "initial_root_height_m": contract.get("initial_root_height_m"),
        "errors": errors,
        "scope": (
            "Portable paths, file presence, 12 revolute joint limits, standing pose, "
            "mass and JSON contract consistency; no CAD sweep, dynamics or hardware test"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / DEFAULT_ASSET_DIR,
        help="Path to training/v67_cts/assets/smalldog_v67",
    )
    args = parser.parse_args()
    report = validate(args.asset_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
