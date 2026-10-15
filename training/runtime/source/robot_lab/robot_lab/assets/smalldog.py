"""V67 shell, fixed head, battery and RS02 actuators with a versioned motion window."""
import json
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(os.environ.get('SMALLDOG_ASSET_DIR', Path(__file__).resolve().parents[4] / 'assets/smalldog_v67'))
CONTRACT = json.loads((ASSET_DIR / 'config/model_contract.json').read_text())
JOINT_NAMES = CONTRACT['joint_order']

SMALLDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_DIR / 'urdf/smalldog_v67.urdf'),
        usd_dir=str(ASSET_DIR / 'usd'),
        usd_file_name=CONTRACT['version'] + '.usd',
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=True,
        collision_from_visuals=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0, angular_damping=0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, CONTRACT['initial_root_height_m']),
        joint_pos=CONTRACT['standing_joint_angles_rad'], joint_vel={'.*': 0.0}),
    soft_joint_pos_limit_factor=CONTRACT['soft_joint_pos_limit_factor'],
    actuators={
        'hip_thigh_rs02': DCMotorCfg(
            joint_names_expr=['.*_(hip|thigh)_joint'],
            effort_limit=7.0, effort_limit_sim=7.0, saturation_effort=17.0,
            velocity_limit=42.9350995991, velocity_limit_sim=42.9350995991,
            stiffness=40.0, damping=0.8, armature=0.0042, friction=0.0),
        'calf_rs02_2to1': DCMotorCfg(
            joint_names_expr=['.*_calf_joint'],
            effort_limit=14.0, effort_limit_sim=14.0, saturation_effort=34.0,
            velocity_limit=21.4675497995, velocity_limit_sim=21.4675497995,
            stiffness=60.0, damping=1.5, armature=0.0168, friction=0.0),
    },
)
