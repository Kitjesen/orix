"""Reuse the CTS observation/reward implementation with SmallDog-specific data."""
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from robot_lab.assets.smalldog import ASSET_DIR, CONTRACT, JOINT_NAMES, SMALLDOG_CFG
from robot_lab.tasks.go2 import mdp as locomotion_mdp
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg


@configclass
class SmallDogEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.scene.env_spacing = 2.5
        self.scene.robot = SMALLDOG_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.terrain.terrain_type = 'plane'
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.visual_material = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.87, 0.9))
        for scanner in (self.scene.height_scanner, self.scene.height_scanner_small):
            scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name='robot', resampling_time_range=(6.0, 10.0),
            rel_standing_envs=0.2, heading_command=False, debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.15, 0.35), lin_vel_y=(-0.10, 0.10), ang_vel_z=(-0.3, 0.3)))
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.scale = {
            f'.*_{kind}_joint': value for kind, value in CONTRACT['action_scale_rad'].items()}
        urdf = ET.parse(ASSET_DIR / 'urdf/smalldog_v67.urdf').getroot()
        self.actions.joint_pos.clip = {}
        for joint in urdf.findall('joint'):
            if joint.get('type') != 'revolute':
                continue
            low, high = (float(joint.find('limit').get(key)) for key in ('lower', 'upper'))
            center = (low + high) / 2.0
            half_width = (high - low) * CONTRACT['soft_joint_pos_limit_factor'] / 2.0
            self.actions.joint_pos.clip[joint.get('name')] = (center - half_width, center + half_width)
        for group in (self.observations.policy, self.observations.critic, self.observations.single_obs):
            for name in ('joint_pos', 'joint_vel', 'joint_acc', 'joint_torque'):
                term = getattr(group, name, None)
                if term is not None:
                    term.params['asset_cfg'] = SceneEntityCfg('robot', joint_names=JOINT_NAMES, preserve_order=True)
        self.observations.critic.contact_force.params['sensor_cfg'] = SceneEntityCfg('contact_forces', body_names='.*_foot_Link')
        for name in ('joint_acc_l2', 'joint_power', 'joint_torques_l2', 'joint_pos_limits'):
            getattr(self.rewards, name).params['asset_cfg'] = SceneEntityCfg('robot', joint_names=JOINT_NAMES)
        self.rewards.base_height_l2.params['asset_cfg'] = SceneEntityCfg('robot', body_names='base_link')
        self.rewards.base_height_l2.params['target_height'] = CONTRACT['base_height_target_m']
        self.rewards.feet_regulation.params['base_height_target'] = CONTRACT['base_height_target_m']
        self.rewards.feet_regulation.params['asset_cfg'] = SceneEntityCfg('robot', body_names='.*_foot_Link')
        self.rewards.undesired_contacts.params['sensor_cfg'] = SceneEntityCfg('contact_forces', body_names='.*_(thigh|calf)_Link')
        self.rewards.flat_orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0)
        # The Go2 0.5 m/s kernel rewarded standing still at SmallDog's low commands.
        self.rewards.track_lin_vel_xy_exp.params['std'] = 0.2
        self.rewards.track_ang_vel_z_exp.params['std'] = 0.35
        self.rewards.feet_air_time = RewardTermCfg(
            func=locomotion_mdp.feet_air_time, weight=0.25,
            params={'command_name': 'base_velocity', 'threshold': 0.12,
                    'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot_Link')})
        self.terminations.illegal_contact.params['sensor_cfg'] = SceneEntityCfg('contact_forces', body_names='base_link')
        self.terminations.bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={'limit_angle': 0.85})
        self.events.randomize_rigid_body_mass_base.params['asset_cfg'] = SceneEntityCfg('robot', body_names='base_link')
        self.events.randomize_rigid_body_mass_base.params['mass_distribution_params'] = (-0.3, 0.3)
        self.events.randomize_com_positions.params['asset_cfg'] = SceneEntityCfg('robot', body_names='base_link')
        self.events.randomize_com_positions.params['com_range'] = {'x': (-0.01, 0.01), 'y': (-0.01, 0.01), 'z': (-0.005, 0.005)}
        self.events.reset_robot_joints.params['position_range'] = (0.99, 1.01)
        self.events.randomize_motor_zero_offset = None
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_material.params.update(
            static_friction_range=(0.7, 1.2), dynamic_friction_range=(0.6, 1.0), restitution_range=(0.0, 0.05))
        self.events.reset_base.params['pose_range'] = {'x': (-0.1, 0.1), 'y': (-0.1, 0.1), 'yaw': (-0.3, 0.3)}
        self.events.reset_base.params['velocity_range'] = {}
        self.curriculum.terrain_levels = None
        self.curriculum.base_linear_velocity = None
        self.curriculum.base_height_l2 = None
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physx.gpu_collision_stack_size = 2**27
        self.episode_length_s = 15
        self.viewer.eye = (1.2, -1.2, 0.9)
        self.viewer.lookat = (0.0, 0.0, 0.32)
        self.viewer.resolution = (1280, 720)
