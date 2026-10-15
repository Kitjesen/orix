"""Build the fixed detachable head from retained native SolidWorks surfaces."""
import gc
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pythoncom
import trimesh
import win32com.client

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
PRIOR = REPO / 'design/v66_three_views_cmf'
sys.path.insert(0, str(REPO / 'smalldog_description/design'))
import build_v30_surfaces as sw

SHIFT = [0, 0, 0]
REMOVED = {'body_tail_cap'}
PORTS = [(x, y) for x in (160, 180) for y in (-24, 24)]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    for folder in ('cad', 'meshes', 'previews', 'process', 'reference'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    spec = json.loads((PRIOR / 'assembly.json').read_text(encoding='utf-8'))
    spec['parts'] = [p for p in spec['parts'] if p['name'] not in REMOVED]
    entries = {p['name']: p for p in spec['parts']}
    for p in spec['parts']:
        for rel in (p['cad'], p['mesh'], str(Path(p['cad']).with_suffix('.step'))):
            shutil.copy2(PRIOR / rel, OUT / rel)
    for name in ('camera_fit.json', 'mount_interfaces.json', 'head_geometry.json', 'preview_parts.json', 'ear_shape.json', 'head_mount_interface.json'):
        shutil.copy2(PRIOR / name, OUT / name)
    shutil.copy2(PRIOR / 'reference/user-reference.png', OUT / 'reference/user-reference.png')
    shutil.copy2(PRIOR / 'cad/frame_reference.SLDPRT', OUT / 'cad/frame_reference.SLDPRT')
    for p in json.loads((PRIOR/'preview_parts.json').read_text())['parts']:
        shutil.copy2(PRIOR/p['mesh'],OUT/p['mesh'])
    record = {'baseline': list(sw.processes().values()), 'output_dir': str(OUT), 'source_files': []}
    app = None
    initial = set()
    write('process/session.json', record)
    try:
        pythoncom.CoInitialize()
        # Discover a PID-specific ROT entry before using the established local COM server.
        rot = pythoncom.GetRunningObjectTable()
        context = pythoncom.CreateBindCtx(0)
        monikers = list(rot.EnumRunning())
        names = [(m, m.GetDisplayName(context, None)) for m in monikers]
        record['solidworks_rot_entries'] = [n for _, n in names if 'sldworks' in n.lower() or 'solidworks' in n.lower()]
        target = record['baseline'][0]['pid'] if len(record['baseline']) == 1 else None
        match = next((m for m, n in names if target and str(target) in n and ('sldworks' in n.lower() or 'solidworks' in n.lower())), None)
        app = win32com.client.Dispatch(rot.GetObject(match)) if match else win32com.client.DispatchEx('SldWorks.Application')
        pid = sw.dynamic(app, 'GetProcessID').GetProcessID()
        bound = sw.processes()[pid]
        if not any(p == bound for p in record['baseline']):
            raise RuntimeError('Unexpected process; stop before editing')
        initial = {sw.value(d, 'GetTitle') for d in (sw.dynamic(app, 'GetDocuments').GetDocuments() or [])}
        record.update(pid=pid, initial_documents=sorted(initial), borrowed_existing_instance=True)
        modeler = sw.value(app, 'GetModeler')
        math = sw.value(app, 'GetMathUtility')

        def identity():
            if sw.processes().get(pid) != bound:
                raise RuntimeError('SolidWorks process identity changed')

        def newdoc():
            identity()
            template = sw.dynamic(app, 'GetUserPreferenceStringValue').GetUserPreferenceStringValue(8)
            return sw.dynamic(app, 'NewDocument').NewDocument(template, 0, 0., 0.)

        def open_doc(path, kind=1):
            identity()
            e, w = sw.integer(), sw.integer()
            d = sw.dynamic(app, 'OpenDoc6').OpenDoc6(str(path), kind, 1, '', e, w)
            if d is None or e.value:
                raise RuntimeError(f'Cannot open {path}: {e.value}')
            record['source_files'].append(str(path))
            return d

        def source(path, select=None):
            d = open_doc(path)
            bodies = list(d.GetBodies2(0, False))
            body = select(bodies) if select else bodies[0]
            result = sw.value(body, 'Copy')
            app.CloseDoc(str(path))
            return result

        def move(body, delta):
            tr = sw.dynamic(math, 'CreateTransform').CreateTransform(sw.array([1,0,0,0,1,0,0,0,1,*[v/1000 for v in delta],1,0,0,0]))
            if not sw.dynamic(body, 'ApplyTransform').ApplyTransform(tr):
                raise RuntimeError('Native translation failed')
            return body

        def box(bounds):
            return sw.primitive(modeler, {'kind': 'box', 'bounds': bounds})

        def cyl(center, radius, length, axis=(0,0,1)):
            return sw.primitive(modeler, {'kind':'cylinder', 'center':center, 'radius':radius, 'length':length, 'axis':axis})

        def deliver(name, body, category=None, density=None):
            d = newdoc()
            f = sw.dynamic(d, 'CreateFeatureFromBody3').CreateFeatureFromBody3(body, False, 0)
            if f is None:
                raise RuntimeError('Body insertion failed: ' + name)
            f.Name = 'V67_' + name
            native = sw.value(sw.value(f, 'GetFaces')[0], 'GetBody')
            m = sw.mesh_body(native, OUT / f'meshes/{name}.stl')
            if not m.is_watertight or len(m.split()) != 1 or m.volume <= 0:
                raise RuntimeError('Invalid native solid: ' + name)
            sw.save(d, OUT / f'cad/{name}.SLDPRT')
            sw.save(d, OUT / f'cad/{name}.step')
            p = entries.setdefault(name, {'name':name, 'cad':f'cad/{name}.SLDPRT', 'mesh':f'meshes/{name}.stl', 'color':'#60717d'})
            p.update(kinematic_group='fixed', category=category or p['category'], bounds_mm=m.bounds.tolist(), volume_mm3=float(m.volume), status='native fit prototype; fastener procurement and structural qualification pending')
            rho = density or p.get('density_assumption_kg_m3', 1270)
            p.update(density_assumption_kg_m3=rho, estimated_solid_mass_kg=float(m.volume*rho*1e-9))
            app.CloseDoc(str(OUT / p['cad']))

        from scipy.interpolate import PchipInterpolator
        from build_v41_body_proportion import BODY, SEAMS, section as original_section
        from native_v38_low_body_bridge import spline_profile
        from build_v33_refinement import HEAD_RADII, head_profile, inner_profile
        from native_v33_refinement import profile
        meta=json.loads((OUT/'head_geometry.json').read_text())
        rows=np.array(meta['parameters']['head_rows'],float)
        rows[0,1:3]=[66,60]
        rows[1,1:3]=[69,63]
        d=newdoc()
        names=[profile(d,f'V67_head_outer_{i}',r[0],head_profile(r),HEAD_RADII) for i,r in enumerate(rows)]
        hout=sw.loft(d,names,'V67_widened_rear_head_outer')
        head_outer_mesh=sw.mesh_body(hout,OUT/'process/head_outer.stl')
        names=[]
        for i,x in enumerate([150,*rows[1:-1,0],254]):
            row=[x,*[float(np.interp(x,rows[:,0],rows[:,j])) for j in range(1,5)]]
            pts,radii=inner_profile(head_profile(row),HEAD_RADII,4)
            names.append(profile(d,f'V67_head_inner_{i}',x,pts,radii))
        hin=sw.loft(d,names,'V67_widened_rear_head_inner')
        sw.save(d,OUT/'process/head_editable_master.SLDPRT')
        shell=sw.operation(hout,hin,sw.SUBTRACT)
        shell=sw.operation(shell,cyl([144,0,150],6,10,(1,0,0)),sw.SUBTRACT)
        for y in [-96,56]:shell=sw.operation(shell,cyl([176,y,163],13.2,40,(0,1,0)),sw.SUBTRACT)
        upper=sw.operation(sw.value(shell,'Copy'),box([146,253.3,-120,120,141.3,230]),sw.INTERSECT)
        lower=sw.operation(shell,box([146,253.3,-120,120,90,140.8]),sw.INTERSECT)
        for x,y in PORTS:
            lower=sw.operation(lower,cyl([x,y,113],5.5,14),sw.UNION)
            lower=sw.operation(lower,cyl([x,y,108],1.7,24),sw.SUBTRACT)
        deliver('head_upper',upper);deliver('head_lower',lower)
        meta['parameters']['head_rows']=rows.tolist()
        meta['parameters']['status']='V67 widened rear head, unchanged face and mounting centers'
        write('head_geometry.json',meta)
        def roof(y):
            cut=head_outer_mesh.section(plane_origin=[146.01,0,0],plane_normal=[1,0,0]);hits=[]
            for line in cut.discrete:
                for a,b in zip(line[:-1],line[1:]):
                    if min(a[1],b[1])<=y<max(a[1],b[1]):hits.append(a[2]+(b[2]-a[2])*(y-a[1])/(b[1]-a[1]))
            return max(hits)
        ys=np.array([0,12,24,36,45,52,57,60,62,63,64],float)
        heights=np.array([roof(y) for y in ys])
        end_roof=PchipInterpolator([*ys,68,72,78,88],[*(heights-.4),135,133,132,132])
        data=np.asarray(BODY,float)
        stations=sorted(set([r[0] for r in BODY if -140<=r[0]<=145]+[-140,-137,-132,-123,-113,30,45,75,85,105,115,137]))
        lift=PchipInterpolator([10,45,75,105,125,145],[0,.055,.22,.52,.78,1])
        rear_top=PchipInterpolator([-140,-132,-123,-112,-75],[130.5,132,133.5,134,129.5])
        def contour(x,offset=0):
            row=[x,*[float(np.interp(x,data[:,0],data[:,j])) for j in range(1,4)]]
            if x<-75:row[3]=float(rear_top(x))
            p=original_section(row,'body',0,128)
            amount=float(lift(np.clip(x,10,145)))
            t=np.clip((p[:,1]-110)/13,0,1)
            target=np.array([float(end_roof(min(abs(y),88))) for y in p[:,0]])
            p[:,1]+=amount*(target-132)*(t*t*(3-2*t))
            if offset:
                tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
                normal=np.column_stack((-tangent[:,1],tangent[:,0]));normal/=np.linalg.norm(normal,axis=1)[:,None]
                p+=offset*normal
            return p
        d=newdoc()
        names=[spline_profile(d,f'V67_body_outer_{i}',x,contour(x)) for i,x in enumerate(stations)]
        outer=sw.loft(d,names,'V67_continuous_body_tail_outer')
        names=[spline_profile(d,f'V67_body_inner_{i}',x,contour(x,3)) for i,x in enumerate([-136.5,*[x for x in stations if -136.5<x<142],142])]
        inner=sw.loft(d,names,'V67_continuous_body_tail_inner')
        sw.save(d,OUT/'process/body_editable_master.SLDPRT')
        # Retain a returned underside in the rear cap area without a roof seam.
        inner=sw.operation(inner,box([-141,-112.9,-160,160,-50,125.5]),sw.SUBTRACT)
        body=sw.operation(outer,inner,sw.SUBTRACT)
        helper=newdoc()
        names=[sw.make_profile(helper,f'V67_split_{i}',x,sw.rectangle(180,-50,z+.6),.5) for i,(x,z) in enumerate(SEAMS)]
        mask=sw.loft(helper,names,'V67_side_seam')
        sw.save(helper,OUT/'process/upper_split.SLDPRT')
        mask=sw.operation(mask,box([-112.9,160,-200,200,-100,240]),sw.INTERSECT)
        mask=sw.operation(mask,box([-145,-112.9,-200,200,-50,122.5]),sw.UNION)
        upper=sw.operation(body,mask,sw.SUBTRACT)
        upper=sw.operation(upper,box([-140,145,-160,160,100,230]),sw.INTERSECT)
        for sx in (-1,1):
            for sy in (-1,1):upper=sw.operation(upper,cyl([sx*126.5,sy*58,60],58.7,112,(sx,0,0)),sw.SUBTRACT)
        upper=sw.operation(upper,box([139,147,-18.6,18.6,109.4,123.6]),sw.SUBTRACT)
        upper=sw.operation(upper,cyl([139,0,150],6,8,(1,0,0)),sw.SUBTRACT)
        deliver('body_upper',upper)
        spec.update(version='V67 continuous upper tail and widened rear head',parts=list(entries.values()),assembly='cad/SD-V67_continuous_head_tail.SLDASM',status='native fit and appearance prototype')
        spec['parameters'].update(revision='V67',source_revision='V66',changed_parts=['body_upper','head_upper','head_lower'],removed_parts=['body_tail_cap'],tail_method='integrated with body upper cover; rear panel remains detachable',head_method='rear upper/lower half-widths 66/60 mm; broadened front shoulder follows head rear roof',head_pose='unchanged low fixed detachable head; face, ear and mount centers retained',tail_root_seam_mm=0)
        write('process/design_change.json',{'changed_parts':['body_upper','head_upper','head_lower'],'removed_part':'body_tail_cap','head_rear_half_widths_mm':[66,60],'head_crown_and_face_pose':'unchanged','tail':'continuous roof and tail in one part; rear service panel retained','limitations':'sampling-based contour matching, not certified G1/G2; head-shell assembly clearances need checking'})
        write('assembly.json',spec);write('native_parts.json',spec)
        for p in spec['parts']:open_doc(OUT/p['cad'])
        open_doc(OUT/'cad/frame_reference.SLDPRT')
        template=sw.dynamic(app,'GetUserPreferenceStringValue').GetUserPreferenceStringValue(9)
        a=sw.dynamic(app,'NewDocument').NewDocument(template,0,0.,0.)
        api=sw.dynamic(a,'AddComponent5','UnfixComponent','FixComponent','ClearSelection2')
        for path,translation in [(OUT/p['cad'],[0,0,0]) for p in spec['parts']]+[(OUT/'cad/frame_reference.SLDPRT',[.130,0,.060])]:
            identity()
            c=api.AddComponent5(str(path),0,'',False,'',0,0,0)
            if c is None:raise RuntimeError('Cannot insert '+path.name)
            tr=sw.dynamic(math,'CreateTransform').CreateTransform(sw.array([1,0,0,0,1,0,0,0,1,*translation,1,0,0,0]))
            c.Select4(False,sw.NULL,False);api.UnfixComponent();c.Transform2=tr
            c.Select4(False,sw.NULL,False);api.FixComponent();api.ClearSelection2(True)
        path=OUT/spec['assembly'];sw.save(a,path);sw.save(a,path.with_suffix('.step'))
        app.CloseDoc(str(path));a=open_doc(path,2)
        refs=[{'name':c.Name2,'path':sw.value(c,'GetPathName'),'suppressed':bool(sw.value(c,'IsSuppressed'))} for c in sw.dynamic(a,'GetComponents').GetComponents(False)]
        if len(refs)!=len(spec['parts'])+1 or not all(Path(c['path']).is_file() and not c['suppressed'] for c in refs):
            raise RuntimeError('Assembly reference check failed')
        write('process/reopen_check.json',{'reopened':True,'components':refs,'component_count':len(refs),'mates':'intentional fixed placement; head articulation removed'})
        print('V67 native assembly saved and reopened',flush=True)
    except Exception:
        record['error']=traceback.format_exc()
        raise
    finally:
        if app is not None:
            docs=list(sw.dynamic(app,'GetDocuments').GetDocuments() or [])
            snapshot=[{'title':sw.value(d,'GetTitle'),'path':sw.value(d,'GetPathName'),'dirty':bool(sw.value(d,'GetSaveFlag'))} for d in docs]
            record['closing_snapshot']=snapshot;record['closed']=[]
            for d in snapshot:
                if d['title'] not in initial and d['path'] and (Path(d['path']).is_relative_to(OUT) or d['path'] in record['source_files']):
                    app.CloseDoc(d['path']);record['closed'].append(d['title'])
            record['remaining_documents']=[sw.value(d,'GetTitle') for d in (sw.dynamic(app,'GetDocuments').GetDocuments() or [])]
            record['left_user_instance_open']=True
        write('process/session.json',record)
        app=None;gc.collect();pythoncom.CoUninitialize()


if __name__=='__main__':
    main()

