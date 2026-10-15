"""Build the V67 simulation asset from saved CAD meshes and mechanical URDF."""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / 'design/v67_continuous_head_tail'
OUT = Path(__file__).parent / 'assets/smalldog_v67'
motion = json.loads((Path(__file__).parent / 'joint_motion_config.json').read_text(encoding='utf-8'))
for sub in ('urdf', 'meshes', 'config'):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
src = Path(__file__).parent / 'source_assets/smalldog_v67_visual_preview.urdf'
root = ET.parse(src).getroot()
root.set('name', motion['version'])
keep = {'base_link', 'battery_guide_link', 'battery_pack_trial_link'}
keep.update(link.get('name') for link in root.findall('link') if link.get('name').startswith(('fr_', 'fl_', 'rr_', 'rl_')))
for link in list(root.findall('link')):
    if link.get('name') not in keep:
        root.remove(link)
for joint in list(root.findall('joint')):
    if joint.find('child').get('link') not in keep:
        root.remove(joint)
    elif joint.get('type') == 'revolute':
        calf = '_calf_' in joint.get('name')
        low, high = motion['joint_position_limits_rad'][joint.get('name')]
        joint.find('limit').set('lower', str(low))
        joint.find('limit').set('upper', str(high))
        joint.find('limit').set('effort', '14' if calf else '7')
        joint.find('limit').set('velocity', str(410 * np.pi / 30 / (2 if calf else 1)))

# All components added here are fixed to base_link. Combine tensors about their
# COM using the parallel-axis theorem, keeping the measured mechanical inertia.
base = root.find("link[@name='base_link']")
i = base.find('inertial')
c = np.fromstring(i.find('origin').get('xyz'), sep=' ')
v = i.find('inertia').attrib
tensor = np.array([[float(v['ixx']), float(v['ixy']), float(v['ixz'])],
                   [float(v['ixy']), float(v['iyy']), float(v['iyz'])],
                   [float(v['ixz']), float(v['iyz']), float(v['izz'])]])
components = [('original_base', float(i.find('mass').get('value')), c, tensor)]
part_records = []
spec = json.loads((DESIGN / 'assembly.json').read_text(encoding='utf-8'))
for part in spec['parts']:
    mesh = trimesh.load(DESIGN / part['mesh'], force='mesh')
    assert mesh.is_watertight and mesh.volume > 0, part['name']
    mesh.apply_scale(0.001)
    mesh.density = part.get('density_assumption_kg_m3', 1270)
    components.append((part['name'], mesh.mass, mesh.center_mass, mesh.moment_inertia))
    part_records.append({'name': part['name'], 'mass_kg': mesh.mass,
                         'density_kg_m3_assumed': mesh.density,
                         'com_m': mesh.center_mass.tolist()})

def box_mass(name, mass, center, size):
    size = np.array(size)
    inertia = np.diag(mass / 12 * (np.sum(size**2) - size**2))
    components.append((name, mass, np.array(center), inertia))

# Payload allowances are explicit estimates until real components are weighed.
box_mass('head_electronics_allowance', 0.2, (0.213, 0, 0.15), (0.08, 0.10, 0.05))
box_mass('compute_board_allowance', 0.15, (0, 0, 0.11), (0.22, 0.12, 0.002))
board = base.find("visual[@name='orix_board_envelope_trial']/origin")
board.set('xyz', '0 0 0.110')
mass = sum(x[1] for x in components)
com = sum(x[1] * x[2] for x in components) / mass
inertia = np.zeros((3, 3))
for _, m, center, local in components:
    d = center - com
    inertia += local + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
i.find('mass').set('value', format(mass, '.12g'))
i.find('origin').set('xyz', ' '.join(format(x, '.12g') for x in com))
for key, (a, b) in dict(ixx=(0, 0), ixy=(0, 1), ixz=(0, 2), iyy=(1, 1), iyz=(1, 2), izz=(2, 2)).items():
    i.find('inertia').set(key, format(inertia[a, b], '.12g'))

def collision_box(link, name, center, size):
    collision = ET.SubElement(link, 'collision', name=name)
    ET.SubElement(collision, 'origin', xyz=' '.join(map(str, center)), rpy='0 0 0')
    ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size=' '.join(map(str, size)))

# Primitive side/belly proxies preserve hip cutouts. Separate convex upper/head
# proxies cover exposed shells without filling the whole shoulder motor space.
collision_box(base, 'v67_belly', (-0.004, 0, 0.009), (0.26, 0.212, 0.031))
for side in (-1, 1):
    collision_box(base, 'v67_side_' + str(side), (-0.004, side * 0.109, 0.07), (0.246, 0.012, 0.105))
for name, selected in [('upper', ['body_upper']), ('head', ['head_upper', 'head_lower', 'head_face']),
                       ('ear_left', ['ear_left']), ('ear_right', ['ear_right'])]:
    meshes = [trimesh.load(DESIGN / ('meshes/' + n + '.stl'), force='mesh') for n in selected]
    hull = trimesh.util.concatenate(meshes).convex_hull
    hull.apply_scale(0.001)
    filename = 'collision_' + name + '.stl'
    hull.export(OUT / 'meshes' / filename)
    collision = ET.SubElement(base, 'collision', name='v67_' + name)
    ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh', filename='../meshes/' + filename)
collision_box(root.find("link[@name='battery_pack_trial_link']"), 'battery', (0, 0, 0), (0.09, 0.145, 0.07))
collision_box(root.find("link[@name='battery_guide_link']"), 'battery_tray', (0, 0, 0.005), (0.102, 0.157, 0.006))

for mesh in root.findall('.//visual/geometry/mesh'):
    relative = Path(mesh.get('filename')).relative_to('..')
    source = OUT / relative
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    # Preserve URDF-relative mesh paths and millimetre scale.

pose = motion['standing_joint_angles_rad']
(OUT / 'config/standing_pose.yaml').write_text(
    ''.join(f'{name}: {value}\n' for name, value in pose.items()), encoding='utf-8')
target = OUT / 'urdf/smalldog_v67.urdf'
root.insert(0, ET.Comment('V67 simulation: geometry-derived shell inertia; estimated payload; sampled joint motion window.'))
ET.indent(root, space='  ')
ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)

# Fail on wrong units, missing assets, invalid inertias, or initial angles
# outside the source mechanical limits before attempting physics import.
for mesh in root.findall('.//geometry/mesh'):
    assert (target.parent / mesh.get('filename')).resolve().is_file()
for link in root.findall('link'):
    node = link.find('inertial/inertia')
    v = {k: float(x) for k, x in node.attrib.items()}
    mat = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']], [v['ixz'], v['iyz'], v['izz']]])
    eig = np.linalg.eigvalsh(mat)
    assert eig[0] > 0 and eig[-1] <= eig[0] + eig[1] + 1e-8, link.get('name')
for j in root.findall('joint'):
    if j.get('type') == 'revolute':
        lim = j.find('limit')
        assert float(lim.get('lower')) <= pose[j.get('name')] <= float(lim.get('upper'))
total = sum(float(n.get('value')) for n in root.findall('./link/inertial/mass'))
contract = {'version': motion['version'], 'shell_source': 'design/v67_continuous_head_tail',
            'mass_kg': total, 'base_mass_kg': mass, 'base_com_m': com.tolist(),
            'base_inertia_kg_m2': inertia.tolist(), 'parts': part_records,
            'joint_order': list(pose), 'standing_joint_angles_rad': pose,
            'joint_position_limits_rad': motion['joint_position_limits_rad'],
            'initial_root_height_m': motion['initial_root_height_m'],
            'base_height_target_m': motion['base_height_target_m'],
            'soft_joint_pos_limit_factor': motion['soft_joint_pos_limit_factor'],
            'action_scale_rad': motion['action_scale_rad'],
            'angle_window_source': motion['angle_window_source'],
            'calf_total_ratio': 2.0, 'ratio_source': 'User confirmed 2:1 on 2026-10-12',
            'motor_internal_ratio_applied_again': False,
            'joint_effort_cap_nm': {'hip': 7, 'thigh': 7, 'calf': 14},
            'joint_speed_cap_rad_s': {'hip': 42.9350995991, 'thigh': 42.9350995991, 'calf': 21.4675497995},
            'estimated_payloads_kg': {'battery': 1.6, 'head_electronics': 0.2, 'compute_board': 0.15},
            'board_center_z_m': 0.11, 'board_relocation_note': 'Simulation packaging proposal: 16 mm lower; mechanical mounting not designed',
            'limitations': ['Sampled simulation joint window; native mechanism and continuous sweep validation remain bounded', 'Material densities and payloads are estimates',
                            'Ideal constant 2:1 calf transmission; no loss, backlash or thermal model',
                            'Fixed head and ears', 'Primitive and convex collision proxies, not full CAD contacts',
                            'PD gains are simulation tuning values; no hardware validation']}
(OUT / 'config/model_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'urdf': str(target), 'mass_kg': total, 'base_mass_kg': mass, 'links': len(root.findall('link')), 'active_joints': len(pose)}))
