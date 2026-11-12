# SmallDog V67 training asset

This directory is the portable asset used by the current SmallDog reinforcement-learning configuration. The model version is `v67_sim_r5_headless`: the headless V67 shell and battery-envelope model with 12 actuated revolute joints.

## Contents

- `urdf/smalldog_v67.urdf` — training URDF with local relative mesh references.
- `meshes/` — 35 referenced STL files for the mechanism, V67 appearance, collision proxies, and trial packaging geometry.
- `config/model_contract.json` — mass, joint order, standing pose, joint limits, actuator caps, payload assumptions, and model limitations.
- `config/standing_pose.yaml` — human-readable copy of the default joint pose.
- `../../joint_motion_config.json` — training-side copy of the motion window and default pose.

Run `python tools/validate_training_asset.py` from the repository root before publishing or after cloning. The script uses only the Python standard library. A relocated asset can be checked with `--asset-dir PATH`.

## Coordinate and unit conventions

- URDF position and geometry values are SI: metres, kilograms, radians, newton-metres, and radians per second.
- Mechanism meshes are already expressed in URDF units where no scale is present.
- V67 appearance meshes and `battery_guide_trial.STL` are authored in millimetres and use `scale="0.001 0.001 0.001"` in the URDF.
- Every URDF mesh reference is relative to this asset directory; no ROS package URI or machine-specific absolute path is required.

## Default pose and recorded model values

- Version: `v67_sim_r5_headless`, derived from `design/v67_continuous_head_tail`.
- Total URDF mass: 13.333464609253827 kg.
- Initial root height: 0.2674918208774965 m.
- Base-height target: 0.2561583499877814 m.
- Default hips: 0 rad.
- Default right thighs: -0.6715534335840251 rad; left thighs: +0.6715534335840251 rad.
- Default right calves: +1.1 rad; left calves: -1.1 rad.

The full named pose, position limits, effort caps, and velocity caps are recorded in `config/model_contract.json`. The validator checks that they match the URDF and `joint_motion_config.json`, and that the standing pose lies inside every hard limit.

## Evidence boundary

The 1.6 kg battery and 0.15 kg compute-board load are simulation packaging estimates. Head electronics (previously 0.2 kg) and the head assembly have been removed. The battery geometry is a trial envelope. The Orix naked-mainboard shape is also a trial visual envelope; it does not confirm the production board, enclosure, connector clearance, mounting, cooling, or assembled hardware mass.

The 2026-11-12 recovery update uses user-requested hip limits of +/-1.05 rad and a common thigh interval [-1.57, 3.49] rad measured from vertical-down zero with positive angles backward. In existing URDF coordinates this becomes [-3.140796326795, 1.919203673205] rad on the right and [-1.919203673205, 3.140796326795] rad on the left. Calf position limits retain the r3 window. All URDF joint effort limits are 17 N.m. These user-configured windows are not measured mechanical stops or a CAD interference clearance validation. The model assumes an ideal constant 2:1 calf transmission and does not model backlash, losses, thermal limits, or a calibrated closed-loop linkage stroke. PD gains, collision proxies, and payload estimates remain simulation assumptions. Passing the validator establishes asset portability and internal contract consistency only; it does not establish CAD interference clearance, dynamics accuracy, or hardware safety.

## Minimum publication set

Publish this complete directory together with:

- `training/v67_cts/joint_motion_config.json`
- the training task and runner configuration that select `v67_sim_r5_headless`
- the policy/checkpoint release notes and configuration needed to interpret any published checkpoint
- `tools/validate_training_asset.py`

Do not publish the URDF without its `meshes/` and `config/` directories.

## Head removal (2026-11-12)

Removed head shells, face, visor, ears, camera/ear decorations, head mounting bridge and fasteners, and the 0.2 kg head electronics allowance. Removed mass: 0.569494988605 kg. Base mass, COM and inertia were rebuilt from the retained components. Body shells, battery, compute board, all leg geometry, joint axes, limits and standing angles remain unchanged. The USD version is new to avoid reusing the headed asset cache.
