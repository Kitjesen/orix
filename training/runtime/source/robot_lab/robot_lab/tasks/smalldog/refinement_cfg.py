"""R4 motion-quality refinement, preserving the original R3 tasks."""
import copy

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import gait_mdp, refinement_mdp
from .gait_cfg import SmallDogGaitEnvCfg, SmallDogGaitRunnerCfg


def _configure_refinement(cfg):
    phase = ObservationTermCfg(
        func=refinement_mdp.gait_phase, params={"command_name": "base_velocity"}
    )
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        group.gait_phase = copy.deepcopy(phase)
    cfg.rewards.diagonal_contact.func = refinement_mdp.diagonal_contact_pattern
    cfg.rewards.swing_clearance.func = refinement_mdp.swing_clearance_reward
    cfg.rewards.excess_swing_clearance = RewardTermCfg(
        func=refinement_mdp.excess_swing_clearance_l2, weight=-30.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"),
                "tolerance": 0.02},
    )
    cfg.rewards.body_pitch_rate = RewardTermCfg(
        func=refinement_mdp.body_pitch_rate_l2, weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards.action_target_clip.weight = -1.0
    cfg.rewards.action_rate_l2.weight = -0.015
    cfg.rewards.action_smoothness_l2.weight = -0.015
    cfg.rewards.base_height_l2.weight = -3.0
    # Commands remain constant within an episode, matching the phase clock contract.
    cfg.commands.base_velocity.resampling_time_range = (15.0, 15.0)


@configclass
class SmallDogRefinedGaitEnvCfg(SmallDogGaitEnvCfg):
    gait_frequency_hz: float = 1.5
    gait_frequency_max_hz: float = 2.5
    gait_frequency_speed_range: tuple[float, float] = (0.3, 1.0)
    gait_stance_fraction: float = 0.60
    gait_clearance_m: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        _configure_refinement(self)


@configclass
class SmallDogRefinedFlatGaitEnvCfg(SmallDogRefinedGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        for name in gait_mdp.FOOT_NAMES:
            getattr(self.scene, f"gait_contact_{name.split('_')[0]}").filter_prim_paths_expr = [
                "/World/ground/terrain/GroundPlane/CollisionPlane"
            ]
        self.curriculum.terrain_levels = None
        self.curriculum.forward_speed = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.rewards.flat_orientation.weight = -4.0
        self.rewards.body_pitch = RewardTermCfg(
            func=refinement_mdp.body_pitch_l2, weight=-4.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_height_l2.weight = -5.0


@configclass
class SmallDogRefinedGaitRunnerCfg(SmallDogGaitRunnerCfg):
    experiment_name = "smalldog_v67_refined_gait_moe_cts"
    max_iterations = 400
    save_interval = 100
