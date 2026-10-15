"""Resolve mesh contact flags using native solid intersection volumes."""
import gc,json,sys
from pathlib import Path
import pythoncom,win32com.client
OUT=Path(__file__).resolve().parents[1];REPO=OUT.parents[1]
sys.path.insert(0,str(REPO/'smalldog_description/design'))
import build_v30_surfaces as sw

baseline=sw.processes();app=None;initial=set();opened=[];result={}
try:
    pythoncom.CoInitialize();app=win32com.client.DispatchEx('SldWorks.Application')
    pid=sw.dynamic(app,'GetProcessID').GetProcessID();bound=sw.processes()[pid]
    if baseline.get(pid)!=bound:raise RuntimeError('Unexpected instance')
    initial={sw.value(d,'GetTitle') for d in (sw.dynamic(app,'GetDocuments').GetDocuments() or [])}
    def read(path):
        if sw.processes().get(pid)!=bound:raise RuntimeError('Process identity changed')
        e,w=sw.integer(),sw.integer();d=sw.dynamic(app,'OpenDoc6').OpenDoc6(str(path),1,1,'',e,w)
        if d is None or e.value:raise RuntimeError('Cannot read '+str(path))
        opened.append(str(path));return [sw.value(b,'Copy') for b in d.GetBodies2(0,False)]
    def intersect(a,b):
        e=sw.integer()
        bodies=sw.dynamic(sw.value(a,'Copy'),'Operations2').Operations2(sw.INTERSECT,sw.value(b,'Copy'),e) or []
        volumes=[float(sw.dynamic(c,'GetMassProperties').GetMassProperties(1000)[3])*1e9 for c in bodies]
        return {'error':e.value,'body_count':len(bodies),'volume_mm3':sum(volumes),'intersection_bounds_mm':[[v*1000 for v in sw.value(c,'GetBodyBox')] for c in bodies]}
    modeler=sw.value(app,'GetModeler')
    def box(bounds):return sw.primitive(modeler,{'kind':'box','bounds':bounds})
    result['controls']={'separated':intersect(box([0,1,0,1,0,1]),box([2,3,0,1,0,1])),
                        'tangent':intersect(box([0,1,0,1,0,1]),box([1,2,0,1,0,1])),
                        'overlap':intersect(box([0,1,0,1,0,1]),box([.5,1.5,0,1,0,1]))}
    report=json.loads((OUT/'process/fit_check.json').read_text())
    pairs=report['mesh_static_contact_candidates']+report['mesh_fastener_contact_candidates']
    bodies={n:read(OUT/f'cad/{n}.SLDPRT')[0] for n in set(n for pair in pairs for n in pair if not n.startswith('frame_'))}
    frame_pairs=[p for p in pairs if p[1].startswith('frame_')]
    if frame_pairs:
        frame=read(OUT/'cad/frame_reference.SLDPRT')
        math=sw.value(app,'GetMathUtility')
        tr=sw.dynamic(math,'CreateTransform').CreateTransform(sw.array([1,0,0,0,1,0,0,0,1,.130,0,.060,1,0,0,0]))
        for i,b in enumerate(frame):
            sw.dynamic(b,'ApplyTransform').ApplyTransform(tr);bodies['native_frame_'+str(i)]=b
        pairs=[p for p in pairs if not p[1].startswith('frame_')]+[[n,'native_frame_'+str(i)] for n in set(p[0] for p in frame_pairs) for i in range(len(frame))]
    result['pairs']=[{'parts':p,**intersect(bodies[p[0]],bodies[p[1]])} for p in pairs]
    result['positive_intersections']=[p for p in result['pairs'] if p['volume_mm3']>1e-5]
    known_empty={result['controls'][k]['error'] for k in ('separated','tangent')}
    result['unresolved']=[p for p in result['pairs'] if not p['body_count'] and p['error'] not in known_empty]
    print(json.dumps(result,indent=2),flush=True)
finally:
    if app is not None:
        docs=list(sw.dynamic(app,'GetDocuments').GetDocuments() or [])
        result['closing_snapshot']=[{'path':sw.value(d,'GetPathName'),'title':sw.value(d,'GetTitle'),'dirty':bool(sw.value(d,'GetSaveFlag'))} for d in docs]
        for d in result['closing_snapshot']:
            if d['title'] not in initial and d['path'] in opened:app.CloseDoc(d['path'])
        result['remaining_documents']=[sw.value(d,'GetTitle') for d in (sw.dynamic(app,'GetDocuments').GetDocuments() or [])]
        result['user_instance_retained']=pid
    (OUT/'process/native_contacts.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    app=None;gc.collect();pythoncom.CoUninitialize()
if result.get('positive_intersections') or result.get('unresolved'):
    raise SystemExit(1)
