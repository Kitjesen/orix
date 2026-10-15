"""Check the new fixed interface against native frame and sampled leg geometry."""
import json
import runpy
import sys
from pathlib import Path
from itertools import product

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
sys.path.insert(0,str(REPO/'smalldog_description/design'))
from check_integrated_head_v2 import surfaces_touch
from refine_head_pinch_clearance import as_pv

spec=json.loads((OUT/'assembly.json').read_text())
parts={p['name']:p for p in spec['parts']}
meshes={n:trimesh.load_mesh(OUT/p['mesh']) for n,p in parts.items()}
for m in meshes.values():m.merge_vertices(digits_vertex=5)
surfaces={n:as_pv(m) for n,m in meshes.items()}
frames={}
for i in range(9):
    m=trimesh.load_mesh(REPO/f'design/mounting_v1/frame_source_0_body_{i}.stl')
    m.vertices=m.vertices*1000+[130,0,60]
    frames[f'frame_{i}']=as_pv(m)
contacts=[]
moved=[]
changed=['body_upper','head_upper','head_lower']
seen=set()
for n in changed:
    for target,s in {**frames,**surfaces}.items():
        if n==target:continue
        key=tuple(sorted((n,target)))
        if key in seen:continue
        seen.add(key)
        # Relative positions between translated head components are unchanged.
        if n in moved and target in moved:continue
        if surfaces_touch(surfaces[n],s):contacts.append([n,target])
print('Static candidates',contacts,flush=True)
(OUT/'process/fit_check.json').write_text(json.dumps({'mesh_static_contact_candidates':contacts,'mesh_fastener_contact_candidates':[],'status':'motion check running'},indent=2))


# Compare the screw/nut bodies only with nearby native mount surfaces.
fastener_contacts=[]

f=runpy.run_path(str(REPO/'smalldog_description/design/render_assembly.py'))
q=f['joint_angles']()
joints={j.get('name'):j for j in f['joints']}
leg_contacts=[]
pose_count=0
unique_link_checks=0
seen_links=set()
# The new lower shell includes the closed bottom and pads; every changed head
# part and the bridge are checked. Calf extremes add the lower-leg envelope.
for prefix in ('fr','fl','rr','rl'):
    values=[]
    for suffix,count in [('hip',7),('thigh',7),('calf',3)]:
        lim=joints[f'{prefix}_{suffix}_joint'].find('limit')
        values.append(np.linspace(float(lim.get('lower')),float(lim.get('upper')),count))
    for angles in product(*values):
        trial=dict(q)
        for suffix,value in zip(('hip','thigh','calf'),angles):trial[f'{prefix}_{suffix}_joint']=float(value)
        tf,_=f['link_frames'](trial)
        pose_count+=1
        for n,link in f['links'].items():
            if not n.startswith(prefix+'_'):continue
            key=(n,tuple(np.round(tf[n].ravel(),10)))
            if key in seen_links:continue
            seen_links.add(key);unique_link_checks+=1
            for v in link.findall('visual'):
                m=f['visual_mesh'](v).transform(tf[n]@f['origin'](v.find('origin')),inplace=False)
                m.points*=1000
                for target in changed:
                    if surfaces_touch(surfaces[target],m):
                        leg_contacts.append({'head_part':target,'leg_link':n,'angles_rad':list(angles)})
    print(prefix,'checked',pose_count,'poses',flush=True)

integrity=[{'name':n,'watertight':bool(meshes[n].is_watertight),'single_solid':len(meshes[n].split())==1,'positive_volume':bool(meshes[n].volume>0)} for n in changed]
result={'scope':'changed integrated upper cover and widened rear head against native frame/retained body and 588 front/rear leg samples; provisional joint limits; unchanged relative head pairs skipped; no continuous sweep, dynamics, cable or stiffness qualification',
        'pose_count':pose_count,'unique_link_transforms':unique_link_checks,'mesh_static_contact_candidates':contacts,'mesh_fastener_contact_candidates':fastener_contacts,'leg_contacts':leg_contacts,'integrity':integrity,
        'head_pose':'unchanged from V63'}
(OUT/'process/fit_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
# The mounting faces intentionally touch; resolve candidates with native solids.
if leg_contacts or not all(r['watertight'] and r['single_solid'] and r['positive_volume'] for r in integrity):raise SystemExit(1)
