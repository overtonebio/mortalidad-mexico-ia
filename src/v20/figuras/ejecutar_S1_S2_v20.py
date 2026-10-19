"""S1/S2 vigentes desde las entradas y resultados del benchmark, sin archivos externos nuevos."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--codigo',type=Path,required=True)
P.add_argument('--config',type=Path,required=True)
P.add_argument('--output-dir',type=Path,required=True)
a=P.parse_args();C=a.codigo.resolve();OUT=a.output_dir.resolve();OUT.mkdir(parents=True)
sys.path.insert(0,str(C))
from estilo_figuras import apply_figure_style
apply_figure_style()
cfg=json.loads(a.config.read_text())
def ns(name):
 p=C/name;g={'__file__':str(p)};exec(compile(p.read_text(),str(p),'exec'),g);return g

def capture(g,fn):
 out=[];g['_save']=lambda fig,outfile:out.append(fig);g[fn](out='x');return out[0]

def numerical(f):
 co=lambda x:np.asarray(x,dtype=float).tolist()
 return [{'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'lines':[(co(x.get_xdata()),co(x.get_ydata())) for x in ax.lines],
          'patches':[(co(p.get_path().vertices),co(p.get_patch_transform().get_matrix())) for p in ax.patches],
          'collections':[(co(z.get_offsets()),[co(v.vertices) for v in z.get_paths()]) for z in ax.collections]} for ax in f.axes]
report=[]
for name,fn in [('S1','figS1b'),('S2','figS2b')]:
 old=capture(ns('figuras_base_v9.py'),fn)
 v19=capture(ns('figuras_base_v9_DP_original.py'),fn)
 new=v19 if name=='S1' else capture(ns('figuras_base_v9_DP_rotulos_20261017.py'),fn)
 assert numerical(old)==numerical(v19)==numerical(new),'Cambio numérico en '+name
 for f in [old,v19,new]:f.canvas.draw()
 box=old.get_tightbbox(old.canvas.get_renderer()).padded(.05);w,h=cfg[name]['output_inches']
 box=Bbox.from_bounds(box.x0+(box.width-w)/2,box.y0+(box.height-h)/2,w,h)
 if name=='S2':
  control=OUT/'control_S2_v19.png';v19.savefig(control,dpi=300,bbox_inches=box,pad_inches=0)
  assert hashlib.sha256(control.read_bytes()).hexdigest()==cfg[name]['control_png_sha256'],'El entorno no reproduce el control S2 v19'
  t=new.get_tightbbox(new.canvas.get_renderer());box=Bbox([[box.x0,box.y0],[box.x1,max(box.y1,t.y1+.05)]])
 for ext in ['pdf','png']:new.savefig(OUT/f'Figura_{name}.{ext}',dpi=300,bbox_inches=box,pad_inches=0)
 report.append({'figura':name,'artistas_numericos_identicos':True,'png_sha256':hashlib.sha256((OUT/f'Figura_{name}.png').read_bytes()).hexdigest()})
 for f in [old,v19,new]:plt.close(f)
(OUT/'verificacion_S1_S2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
