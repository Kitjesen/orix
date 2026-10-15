"""R6 rough terrain task with high-speed gait extension."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from .refinement_cfg import SmallDogRefinedGaitEnvCfg, SmallDogRefinedGaitRunnerCfg
from .rough_cfg import ForwardSpeedCurriculum


@configclass
class SmallDogTerrainR6EnvCfg(SmallDogRefinedGaitEnvCfg):
    """Warm-start terrain training from levels zero through ten."""

    gait_frequency_high_speed_hz: float = 3.0
    gait_frequency_high_speed_m_s: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_generator.num_rows = 11
        self.scene.terrain.max_init_terrain_level = 10
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 1.0)
        self.curriculum.terrain_levels = CurriculumTermCfg(
            func=self.curriculum.terrain_levels.func
        )
        self.curriculum.forward_speed = CurriculumTermCfg(
            func=ForwardSpeedCurriculum,
            params={"max_speed": 1.5, "increment": 0.1, "batch_size": 128,
                    "success_ratio": 0.7},
        )


@configclass
class SmallDogTerrainR6RunnerCfg(SmallDogRefinedGaitRunnerCfg):
    experiment_name = "smalldog_v67_refined_gait_moe_cts"
    max_iterations = 10000
    save_interval = 100
