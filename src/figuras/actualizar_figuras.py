"""Actualiza solo rótulos; conserva todos los datos y artistas de las figuras originales.
Uso: python actualizar_figuras.py --benchmark /ruta/benchmark --out /ruta/figuras
"""
from pathlib import Path
import argparse,os,sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
from estilo_figuras import apply_figure_style
p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
os.chdir(args.benchmark.resolve());apply_figure_style()
g={'__file__':str(D/'figuras_base_v9.py')};exec((D/'figuras_base_v9.py').read_text(),g)
def data_hash(fig):
 data=[]
 for ax in fig.axes:
  def vals(v):return np.asarray(v).tolist()
  data.append({'lines':[{'x':vals(x.get_xdata()),'y':vals(x.get_ydata())} for x in ax.lines],
    'patches':[vals(x.get_path().vertices) for x in ax.patches],
    'patch_extents':[vals(x.get_extents().get_points()) for x in ax.patches],
    'collections':[vals(x.get_offsets()) for x in ax.collections],
    'xlim':ax.get_xlim(),'ylim':ax.get_ylim()})
 return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
audit={}
def save(fig,out):
 before=data_hash(fig);name=Path(out).stem
 if name=='Figura_6':
  repl={'Ajuste completo, 1 000 pasos':'Chronos-2, ajuste completo',
   'LoRA, 300 pasos':'Chronos-2-LoRA, 300 pasos',
   'LoRA, 1 000 pasos (3 semillas)':'Chronos-2-LoRA, 1 000 pasos†',
   'LoRA, sin cáncer de mama':'Chronos-2-LoRA, sin mama',
   'LoRA, exclusión estricta*':'Chronos-2-LoRA, exclusión estricta*',
   'WQL vs Chronos-2 sin adaptar (%)':'WQL vs Chronos-2 sin adaptación (%)',
   'WQL vs sin adaptar (%)':'WQL vs sin adaptación (%)',
   'Sin\nadaptar':'Sin\nadaptación','Todas':'LoRA\nTodas',
   'Sin\nmama':'LoRA\nSin mama','Estricta*':'LoRA\nEstricta*'}
  for t in fig.findobj(match=Text):
   if t.get_text() in repl:t.set_text(repl[t.get_text()])
  fig.axes[0].set_yticklabels([repl.get(t.get_text(),t.get_text()) for t in fig.axes[0].get_yticklabels()])
  fig.axes[2].set_xticklabels(['Original','LoRA\ntodas','LoRA\nsin mama','LoRA\nestricta*'],fontsize=5.5)
  for t in fig.texts:
   if t.get_text().startswith('* Sin series'):
    t.set_text('Ajuste completo: 1 000 pasos. † Barra y porcentaje: semilla 0; puntos: semillas 0, 1 y 2.\n* Sin series de cáncer de mama ni de todas las causas en el entrenamiento.')
    t.set_position((0.01,-0.09))
 elif name=='Figura_S3':
  for t in fig.findobj(match=Text):
   if t.get_text()=='WQL vs zero-shot (%)':t.set_text('WQL vs sin adaptación (%)')
  for t in fig.axes[2].get_legend().get_texts():
   if t.get_text()=='Zero-shot':t.set_text('Sin adaptación')
   elif t.get_text()=='LoRA':t.set_text('Chronos-2-LoRA')
  fig.text(0.01,-0.07,'Piloto de Chronos-2. LoRA: Chronos-2-LoRA del piloto; contexto: información entre series.',fontsize=6.5)
 after=data_hash(fig);assert before==after,'Se modificaron elementos numéricos'
 audit[name]={'datos_antes_sha256':before,'datos_despues_sha256':after,'identicos':True}
 fig.savefig(out,dpi=300,bbox_inches='tight',pad_inches=.05)
 fig.savefig(Path(out).with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
g['_save']=save
G=json.loads(Path('entradas/numeros_G.json').read_text())
g['fig3b']({'LightGBM global':G['lgbm_same'],'Ensamble (mediana)':G['ens_same']},out=str(args.out/'Figura_6.pdf'))
g['figS3b'](out=str(args.out/'Figura_S3.pdf'))
(args.out/'validacion_datos_figuras.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,indent=2))
