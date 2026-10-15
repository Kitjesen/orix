"""Render saved CAD and URDF geometry with an explicit CMF and orthographic cameras."""
import json, runpy, sys
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
OUT=Path(__file__).resolve().parents[1]; REPO=OUT.parents[1]
sys.path.insert(0,str(REPO/'smalldog_description/design'))
import render_v41_body_proportion as base
import render_v46_head_body_gimbal as current
CMF={'main':'#ecebe5','lower':'#39454c','joint':'#59646c','thigh':'#b6c0c7','accent':'#39b8d6','foot':'#182128'}
original=base.part_color
def color(part):
    n=part['name'];c=part.get('category')
    if n=='body_lower':return CMF['lower']
    if n=='head_frame_bridge':return CMF['lower']
    if c in ('body_shell','head_shell','ear_shell'):return CMF['main']
    return original(part)
base.part_color=color
class Scene(base.Scene):
    def add(self,mesh,color,group='fixed',category=None,lighting=True):
        if category is None:
            color={'#515c67':CMF['joint'],'#a5adb6':CMF['thigh'],'#141b22':CMF['foot']}.get(color,color)
        if category=='body_shell' and mesh.bounds[5]>70 and mesh.bounds[2]*mesh.bounds[3]>0:
            top=mesh.clip(normal=(0,0,1),origin=(0,0,46),invert=False)
            bottom=mesh.clip(normal=(0,0,1),origin=(0,0,46),invert=True)
            for piece,c in [(top,[236,235,229]),(bottom,[57,69,76])]:
                piece.point_data['paint_rgb']=np.tile(c,(piece.n_points,1)).astype(np.uint8)
            actor=self.plotter.add_mesh(top.merge(bottom,merge_points=False),scalars='paint_rgb',rgb=True,smooth_shading=True,split_sharp_edges=True,ambient=.3,diffuse=.68,specular=.18,specular_power=24)
            self.body_actors.append(actor)
        else:actor=super().add(mesh,color,group,category,lighting)
        if not lighting and category is None:self.ground=actor
        if category in ('body_shell','head_shell','ear_shell'):
            actor.GetProperty().SetSpecular(.18)
        return actor
    def pose(self,*args):self.plotter.render()
    def graphics(self,graphics):
        spec=json.loads((self.directory/'ear_shape.json').read_text())
        graphics['ear_inset_xz_mm']=spec['inset_xz_mm']
        graphics['ear_accent_xz_mm']=spec['accent_xz_mm']
        super().graphics(graphics)
mechanical=runpy.run_path(str(REPO/'smalldog_description/design/render_assembly.py'))
camera=json.loads((OUT/'camera_fit.json').read_text())
s=Scene(OUT,mechanical,camera)
s.ground.SetVisibility(False)
s.screenshot(OUT/'previews/01-assembled.png')
s.ground.SetVisibility(False)
views={
 'side': ([(20,-1000,-40),(20,0,-40),(0,0,1)],'SIDE / head to +X'),
 'front': ([(1000,0,-40),(0,0,-40),(0,0,1)],'FRONT / look along -X'),
 'top': ([(20,0,1000),(20,0,0),(0,1,0)],'TOP / head to +X')}
for name,(position,label) in views.items():
    s.plotter.camera_position=position;s.plotter.camera.parallel_projection=True
    s.plotter.camera.parallel_scale=300;s.plotter.camera.SetWindowCenter(0,0)
    s.screenshot(OUT/f'previews/02-{name}.png')
s.ground.SetVisibility(True)
for name in ('rear','tail','head-neck'):
    current.set_view(s,name);s.screenshot(OUT/f'previews/03-{name}.png')
s.plotter.close()
# Identical material, camera and pose isolate the tail geometry revision.
for directory,label in [(REPO/'design/v66_three_views_cmf','before'),(OUT,'after')]:
    s=Scene(directory,mechanical,camera)
    current.set_view(s,'tail');s.screenshot(OUT/f'previews/04-tail-{label}.png')
    s.plotter.close()
s=Scene(REPO/'design/v66_three_views_cmf',mechanical,camera)
s.ground.SetVisibility(False)
for name,(position,label) in views.items():
    s.plotter.camera_position=position;s.plotter.camera.parallel_projection=True
    s.plotter.camera.parallel_scale=300;s.plotter.camera.SetWindowCenter(0,0)
    s.screenshot(OUT/f'previews/06-before-{name}.png')
s.plotter.close()
# Contact sheet combines CAD renders without geometric image manipulation.
font='C:/Windows/Fonts/msyh.ttc'
def f(size):return ImageFont.truetype(font,size)
board=Image.new('RGB',(1840,1400),'#f8f9fa');draw=ImageDraw.Draw(board)
draw.text((40,22),'V67  /  三视图与配色检查',font=f(34),fill='#243440')
draw.text((40,70),'实际 CAD + URDF 腿部 · 固定可拆头部 · 暖白 / 石墨灰 / 青色',font=f(20),fill='#596775')
for file,title,x,y in [('02-side.png','侧视图  +X 前方 →',20,126),('02-front.png','正视图',940,126),('02-top.png','俯视图  +X 前方 →',20,742),('01-assembled.png','整机效果（此格不作比例比较）',940,742)]:
    draw.text((x+20,y),title,font=f(24),fill='#243440')
    im=Image.open(OUT/'previews'/file).convert('RGB').resize((900,562),Image.Resampling.LANCZOS)
    board.paste(im,(x,y+42))
    if file.startswith('02'):
        # 900px source height / 600mm, scaled by 900/1440.
        a=x+36;b=a+94;z=y+573
        draw.line((a,z,b,z),fill='#243440',width=3)
        draw.line((a,z-5,a,z+5),fill='#243440',width=2);draw.line((b,z-5,b,z+5),fill='#243440',width=2)
        draw.text((a,z-30),'100 mm',font=f(17),fill='#243440')
draw.text((40,1360),'三视图统一正交比例；仅供造型与包络评审，不是加工图。尾盖和装配源文件保留。',font=f(18),fill='#596775')
board.save(OUT/'previews/00-three-views.png')
(OUT/'process/render_setup.json').write_text(json.dumps({'cmf_hex':CMF,'orthographic':True,'three_view_parallel_scale_mm':300,'image_size_px':[1440,900],'scale_px_per_mm':1.5,'cameras':views,'unchanged_geometry':['face','ears','frame','motors','legs','body_sides'],'appearance_scope':'render material study, not manufacturing coating specification','contact_sheet':'uniform resize, no geometry distortion'},ensure_ascii=False,indent=2),encoding='utf-8')
print('V67 CAD orthographics, CMF hero and tail comparisons saved',flush=True)
