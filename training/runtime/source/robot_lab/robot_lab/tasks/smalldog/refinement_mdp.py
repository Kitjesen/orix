"""R4 gait timing and motion-quality terms for SmallDog."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

from .gait_mdp import (
    action_target_clip_l1,
    foot_clearance,
    foot_contact_data,
    moving_mask,
    phase_contact_score,
)


def gait_frequency_from_command(
    command: torch.Tensor,
    frequency_min_hz: float,
    frequency_max_hz: float,
    speed_range: tuple[float, float],
) -> torch.Tensor:
    """Map planar command speed to gait frequency with bounded linear scaling."""
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    speed_min, speed_max = speed_range
    blend = ((speed - speed_min) / (speed_max - speed_min)).clamp(0.0, 1.0)
    return frequency_min_hz + blend * (frequency_max_hz - frequency_min_hz)


def gait_frequency(env, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    frequency = gait_frequency_from_command(
        command,
        env.cfg.gait_frequency_hz,
        env.cfg.gait_frequency_max_hz,
        env.cfg.gait_frequency_speed_range,
    )
    # R6 can extend only the high-speed tail; older tasks have no extension
    # attributes and therefore retain the original 1.5-2.5 Hz curve.
    high_speed_hz = getattr(env.cfg, "gait_frequency_high_speed_hz", None)
    high_speed_m_s = getattr(env.cfg, "gait_frequency_high_speed_m_s", None)
    if high_speed_hz is None or high_speed_m_s is None:
        return frequency
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    base_speed = env.cfg.gait_frequency_speed_range[1]
    high_blend = ((speed - base_speed) / (high_speed_m_s - base_speed)).clamp(0.0, 1.0)
    high_frequency = env.cfg.gait_frequency_max_hz + high_blend * (
        high_speed_hz - env.cfg.gait_frequency_max_hz
    )
    return torch.where(speed > base_speed, high_frequency, frequency)


def _cycle_phase(env, command_name: str = "base_velocity") -> torch.Tensor:
    elapsed = env.episode_length_buf * env.step_dt
    return (elapsed * gait_frequency(env, command_name)).remainder(1.0)


def gait_phase(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Expose the exact speed-dependent clock used by the R4 reward targets."""
    angle = _cycle_phase(env, command_name) * (2.0 * math.pi)
    phase = torch.stack((angle.sin(), angle.cos()), dim=-1)
    return phase * moving_mask(env, command_name).unsqueeze(-1)


def gait_targets(
    env, command_name: str = "base_velocity"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build diagonal stance and clearance targets from the shared R4 clock."""
    phase = _cycle_phase(env, command_name)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=env.device)
    foot_phase = (phase.unsqueeze(-1) + offsets).remainder(1.0)
    duty = env.cfg.gait_stance_fraction
    stance = foot_phase < duty
    swing_progress = ((foot_phase - duty) / (1.0 - duty)).clamp(0.0, 1.0)
    clearance = env.cfg.gait_clearance_m * torch.sin(math.pi * swing_progress)
    distance_to_zero = torch.minimum(foot_phase, 1.0 - foot_phase)
    stable = (distance_to_zero > 0.025) & ((foot_phase - duty).abs() > 0.025)
    return stance, clearance, stable


def diagonal_contact_pattern(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contacts, valid, _ = foot_contact_data(env, asset_cfg, sensor_cfg)
    contacts = contacts & valid
    expected, _, stable = gait_targets(env, command_name)
    score = phase_contact_score(contacts, expected)
    positive = score.clamp_min(0.0) * (-asset.data.projected_gravity_b[:, 2]).clamp(
        0.0, 1.0
    )
    return (
        (positive + score.clamp_max(0.0))
        * stable.all(dim=-1)
        * moving_mask(env, command_name)
    )


def swing_clearance_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.02,
) -> torch.Tensor:
    expected_stance, target, _ = gait_targets(env, command_name)
    clearance, valid = foot_clearance(env, asset_cfg)
    score = torch.exp(-((clearance - target) / std).square())
    swing = ~expected_stance
    result = (score * swing * valid).sum(dim=-1) / swing.sum(dim=-1).clamp_min(1)
    upright = (-env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]).clamp(
        0.0, 1.0
    )
    return result * upright * moving_mask(env, command_name)


def excess_swing_clearance(
    clearance: torch.Tensor,
    target: torch.Tensor,
    expected_stance: torch.Tensor,
    valid: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Average squared excess over valid feet expected to be in swing."""
    swing_valid = (~expected_stance) & valid
    excess = torch.where(
        swing_valid,
        (clearance - target - tolerance).clamp_min(0.0),
        torch.zeros_like(clearance),
    )
    return excess.square().sum(dim=-1) / swing_valid.sum(dim=-1).clamp_min(1)


def excess_swing_clearance_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.02,
) -> torch.Tensor:
    expected_stance, target, _ = gait_targets(env, command_name)
    clearance, valid = foot_clearance(env, asset_cfg)
    penalty = excess_swing_clearance(
        clearance, target, expected_stance, valid, tolerance
    )
    return penalty * moving_mask(env, command_name)


def body_pitch_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize global body pitch using the body-frame gravity x component."""
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 0].square()


def body_pitch_rate_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize body-frame pitch angular velocity."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 1].square()


__all__ = [
    "action_target_clip_l1",
    "body_pitch_l2",
    "body_pitch_rate_l2",
    "diagonal_contact_pattern",
    "excess_swing_clearance_l2",
    "gait_frequency",
    "gait_frequency_from_command",
    "gait_phase",
    "gait_targets",
    "swing_clearance_reward",
]
