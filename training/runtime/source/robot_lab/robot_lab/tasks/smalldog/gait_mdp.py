"""SmallDog foot geometry and observable periodic gait shaping."""
from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import raycast_mesh

FOOT_NAMES = ("fl_foot_Link", "fr_foot_Link", "rl_foot_Link", "rr_foot_Link")
# These values are checked against all four URDF foot collision shapes before training.
FOOT_SPHERE_OFFSET = (0.01658312383, 0.0, 0.0)
FOOT_SPHERE_RADIUS = 0.032


def _foot_ids(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    indices = getattr(env, "_smalldog_foot_indices", None)
    if indices is None:
        indices = env._smalldog_foot_indices = {}
    key = (asset_cfg.name, sensor_cfg.name)
    if key not in indices:
        asset_ids, names = asset.find_bodies(FOOT_NAMES, preserve_order=True)
        # ContactSensor.body_names reads all replicated prim paths on every access.
        sensor_names = sensor.body_names
        sensor_ids = [sensor_names.index(name) for name in names]
        indices[key] = asset_ids, sensor_ids
    asset_ids, sensor_ids = indices[key]
    return asset, sensor, asset_ids, sensor_ids


def moving_mask(env, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.1) | (command[:, 2].abs() > 0.1)


def gait_phase(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Expose the same per-environment clock used by the reward targets."""
    phase = env.episode_length_buf * env.step_dt * env.cfg.gait_frequency_hz
    angle = phase * (2.0 * math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1) * moving_mask(env, command_name).unsqueeze(-1)


def gait_targets(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FL/RR alternate with FR/RL, with intentional double support at low speed."""
    phase = (env.episode_length_buf * env.step_dt * env.cfg.gait_frequency_hz).remainder(1.0)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=env.device)
    foot_phase = (phase.unsqueeze(-1) + offsets).remainder(1.0)
    duty = env.cfg.gait_stance_fraction
    stance = foot_phase < duty
    swing_progress = ((foot_phase - duty) / (1.0 - duty)).clamp(0.0, 1.0)
    clearance = env.cfg.gait_clearance_m * torch.sin(math.pi * swing_progress)
    distance_to_zero = torch.minimum(foot_phase, 1.0 - foot_phase)
    stable = (distance_to_zero > 0.025) & ((foot_phase - duty).abs() > 0.025)
    return stance, clearance, stable


def contact_point_velocity(com_velocity, angular_velocity, contact_position, com_position):
    """Rigid-body velocity at the measured contact point, including rotation."""
    return com_velocity + torch.linalg.cross(angular_velocity, contact_position - com_position, dim=-1)


def foot_contact_data(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    asset, sensor, asset_ids, sensor_ids = _foot_ids(env, asset_cfg, sensor_cfg)
    points = torch.stack([
        env.scene.sensors[f"gait_contact_{name.split('_')[0]}"].data.contact_pos_w[:, 0, 0]
        for name in FOOT_NAMES
    ], dim=1)
    contacts = sensor.data.net_forces_w[:, sensor_ids].norm(dim=-1) > 1.0
    valid = torch.isfinite(points).all(dim=-1)
    # Non-contact positions are NaN; replace them before the cross product.
    points = torch.where(valid.unsqueeze(-1), points, asset.data.body_com_pos_w[:, asset_ids])
    velocity = contact_point_velocity(
        asset.data.body_com_lin_vel_w[:, asset_ids],
        asset.data.body_com_ang_vel_w[:, asset_ids],
        points, asset.data.body_com_pos_w[:, asset_ids],
    )
    return contacts, valid, velocity


def feet_contact_slip_l2(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize actual ground contact point horizontal velocity while supporting."""
    contacts, valid, velocity = foot_contact_data(env, asset_cfg, sensor_cfg)
    return (velocity[..., :2].square().sum(dim=-1) * contacts * valid).sum(dim=-1)


def foot_clearance(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertical sphere-bottom clearance above terrain sampled at each sphere center XY.

    This is vertical clearance, not the minimum normal gap to a sloped surface.
    The ground query uses the existing terrain mesh, with four rays per environment.
    """
    asset = env.scene[asset_cfg.name]
    ids, _ = asset.find_bodies(FOOT_NAMES, preserve_order=True)
    position = asset.data.body_link_pos_w[:, ids]
    offset = torch.tensor(FOOT_SPHERE_OFFSET, device=env.device).expand_as(position)
    center = position + quat_apply(asset.data.body_link_quat_w[:, ids], offset)
    starts = center.clone()
    starts[..., 2] += 1.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    scanner = env.scene.sensors["height_scanner"]
    ground = raycast_mesh(starts, directions, scanner.meshes[scanner.cfg.mesh_prim_paths[0]], max_dist=3.0)[0]
    valid = torch.isfinite(ground).all(dim=-1)
    clearance = center[..., 2] - FOOT_SPHERE_RADIUS - ground[..., 2]
    return torch.where(valid, clearance, 0.0), valid


def action_target_clip_l1(env, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Penalize position targets that the action term clips to URDF limits."""
    term = env.action_manager.get_term(action_term_name)
    unclipped = term.raw_actions * term._scale + term._offset
    return (unclipped - term.processed_actions).abs().sum(dim=-1)


def phase_contact_score(contacts: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    mismatch = (contacts != expected).float().mean(dim=-1)
    # Partial matches get a penalty. A fixed pair cannot collect half-cycle bonuses.
    return (mismatch == 0.0).float() - mismatch


def diagonal_contact_pattern(env, command_name: str, asset_cfg: SceneEntityCfg,
                             sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contacts, valid, _ = foot_contact_data(env, asset_cfg, sensor_cfg)
    contacts = contacts & valid
    expected, _, stable = gait_targets(env)
    score = phase_contact_score(contacts, expected)
    # Timing tolerance is explicit; sustained three-foot support receives no bonus.
    positive = score.clamp_min(0.0) * (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    return (positive + score.clamp_max(0.0)) * stable.all(dim=-1) * moving_mask(env, command_name)


def swing_clearance_reward(env, command_name: str, asset_cfg: SceneEntityCfg,
                           std: float = 0.02) -> torch.Tensor:
    expected_stance, target, _ = gait_targets(env)
    clearance, valid = foot_clearance(env, asset_cfg)
    score = torch.exp(-((clearance - target) / std).square())
    swing = ~expected_stance
    result = (score * swing * valid).sum(dim=-1) / swing.sum(dim=-1).clamp_min(1)
    upright = (-env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    return result * upright * moving_mask(env, command_name)
