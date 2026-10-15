"""SmallDog terrain and speed curricula; retains the validated V67 articulation."""
import json

import isaaclab.terrains as terrain_gen
import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel

from .env_cfg import SmallDogEnvCfg
from .rsl_rl_cfg import SmallDogMoECTSRunnerCfg


def terrain_levels_after_episode(env, env_ids):
    """Do not score uninitialized root poses during the first reset."""
    ids = torch.as_tensor(env_ids, device=env.device)
    ids = ids[env.episode_length_buf[ids] > 0]
    if ids.numel():
        return terrain_levels_vel(env, ids)
    return env.scene.terrain.terrain_levels.float().mean()


class ForwardSpeedCurriculum(ManagerTermBase):
    """Raise the command ceiling only after successful forward-command episodes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = 0
        self.passed = 0
        self.error_sum = 0.0
        self.completed = 0

    def __call__(self, env, env_ids, max_speed=1.0, increment=0.1, batch_size=128, success_ratio=0.7):
        command = env.command_manager.get_term('base_velocity')
        ceiling = command.cfg.ranges.lin_vel_x[1]
        # The runner randomizes initial episode timers; discard that partial startup batch.
        if env.common_step_counter < env.max_episode_length:
            return ceiling
        ids = torch.as_tensor(env_ids, device=env.device)
        cmd = command.command[ids]
        # Each command lasts a full episode, so its final tracking metric is meaningful.
        eligible = ((cmd[:, 0] >= 0.6 * ceiling)
                    & (cmd[:, 1].abs() < 0.06) & (cmd[:, 2].abs() < 0.15)
                    & (env.episode_length_buf[ids] > 0))
        selected = ids[eligible]
        self.count += int(selected.numel())
        if selected.numel():
            tolerance = torch.clamp(command.command[selected, 0] * 0.25, min=0.08)
            passed = (env.reset_time_outs[selected]
                      & (command.metrics['error_vel_xy'][selected] < tolerance))
            self.passed += int(passed.sum())
            self.completed += int(env.reset_time_outs[selected].sum())
            self.error_sum += float(command.metrics['error_vel_xy'][selected].sum())
        if self.count >= batch_size:
            print('SMALLDOG_SPEED_GATE ' + json.dumps({
                'control_step': int(env.common_step_counter), 'ceiling_m_s': ceiling,
                'eligible_episodes': self.count, 'passed_episodes': self.passed,
                'pass_ratio': self.passed / self.count,
                'timeout_ratio': self.completed / self.count,
                'mean_normalized_xy_error_m_s': self.error_sum / self.count,
                'required_pass_ratio': success_ratio,
                'advance': self.passed / self.count >= success_ratio and ceiling < max_speed,
            }), flush=True)
            if self.passed / self.count >= success_ratio:
                ceiling = min(max_speed, round(ceiling + increment, 6))
                command.cfg.ranges.lin_vel_x = (-0.15, ceiling)
            self.count = 0
            self.passed = 0
            self.error_sum = 0.0
            self.completed = 0
        return ceiling


@configclass
class SmallDogRoughEnvCfg(SmallDogEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=42, size=(6.0, 6.0), border_width=10.0,
            num_rows=8, num_cols=20, horizontal_scale=0.1, vertical_scale=0.005,
            slope_threshold=0.75, curriculum=True, use_cache=False,
            sub_terrains={
                'flat': terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
                'rough': terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.15, noise_range=(0.005, 0.025), noise_step=0.005,
                    downsampled_scale=0.2, border_width=0.25),
                'slope_up': terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.10, slope_range=(0.0, 0.22), platform_width=1.5, border_width=0.25),
                'slope_down': terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.10, slope_range=(0.0, 0.22), platform_width=1.5, border_width=0.25),
                'stairs_up': terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.10, step_height_range=(0.01, 0.10), step_width=0.25,
                    platform_width=1.5, border_width=0.5, holes=False),
                'stairs_down': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.10, step_height_range=(0.01, 0.10), step_width=0.25,
                    platform_width=1.5, border_width=0.5, holes=False),
                'obstacles': terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.10, obstacle_width_range=(0.3, 0.8),
                    obstacle_height_range=(0.01, 0.10), num_obstacles=20,
                    platform_width=1.5, border_width=0.25),
                'wave': terrain_gen.HfWaveTerrainCfg(
                    proportion=0.05, amplitude_range=(0.01, 0.08), num_waves=3, border_width=0.25),
            })
        self.commands.base_velocity.resampling_time_range = (15.0, 15.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_levels_after_episode)
        self.curriculum.forward_speed = CurriculumTermCfg(
            func=ForwardSpeedCurriculum,
            params={'max_speed': 1.0, 'increment': 0.1, 'batch_size': 128, 'success_ratio': 0.7})
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**29


@configclass
class SmallDogRoughRunnerCfg(SmallDogMoECTSRunnerCfg):
    experiment_name = 'smalldog_v67_moe_cts'
    max_iterations = 3000
