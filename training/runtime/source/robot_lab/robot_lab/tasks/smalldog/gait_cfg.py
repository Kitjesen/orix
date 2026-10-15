"""Observable periodic gait tasks, with a separate low-speed flat pilot."""
import copy

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from robot_lab.tasks.go2.env_cfg import RewardsCfg

from . import gait_mdp
from .rough_cfg import SmallDogRoughEnvCfg, SmallDogRoughRunnerCfg


@configclass
class SmallDogGaitRewardsCfg(RewardsCfg):
    feet_regulation = RewardTermCfg(
        func=gait_mdp.feet_contact_slip_l2, weight=-0.50,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_Link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_Link")},
    )
    action_target_clip = RewardTermCfg(
        func=gait_mdp.action_target_clip_l1, weight=-0.50,
        params={"action_term_name": "joint_pos"},
    )
    diagonal_contact = RewardTermCfg(
        func=gait_mdp.diagonal_contact_pattern, weight=0.75,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_Link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_Link")},
    )
    swing_clearance = RewardTermCfg(
        func=gait_mdp.swing_clearance_reward, weight=0.50,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "std": 0.02},
    )


@configclass
class SmallDogGaitEnvCfg(SmallDogRoughEnvCfg):
    gait_frequency_hz: float = 1.5
    gait_stance_fraction: float = 0.65
    gait_clearance_m: float = 0.04

    def __post_init__(self):
        self.rewards = SmallDogGaitRewardsCfg()
        super().__post_init__()
        self.rewards.feet_regulation.params.pop("base_height_target", None)
        # Phase and clearance terms now define timing; keep airtime as a weak auxiliary.
        self.rewards.feet_air_time.params["threshold"] = 0.08
        self.rewards.feet_air_time.weight = 0.10
        phase = ObservationTermCfg(func=gait_mdp.gait_phase, params={"command_name": "base_velocity"})
        for group in (self.observations.policy, self.observations.single_obs, self.observations.critic):
            group.gait_phase = copy.deepcopy(phase)
        # Isaac's filtered contact reporting requires one body per sensor.
        for name in gait_mdp.FOOT_NAMES:
            setattr(self.scene, f"gait_contact_{name.split('_')[0]}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}", track_contact_points=True,
                filter_prim_paths_expr=["/World/ground/terrain/mesh"],
                max_contact_data_count_per_prim=4,
            ))


@configclass
class SmallDogFlatGaitEnvCfg(SmallDogGaitEnvCfg):
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.30)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.10


@configclass
class SmallDogGaitRunnerCfg(SmallDogRoughRunnerCfg):
    experiment_name = "smalldog_v67_gait_moe_cts"
    max_iterations = 600
