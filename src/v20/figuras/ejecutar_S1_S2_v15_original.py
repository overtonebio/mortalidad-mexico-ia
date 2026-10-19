"""Reproduce solo S1/S2 desde resultados guardados; no vuelve a ajustar modelos."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True,help='Carpeta nueva fuera de la adenda.')
    p.add_argument('--prepare-only',action='store_true',help='Comprueba y copia insumos, sin renderizar figuras.')
    a=p.parse_args();out=a.output_dir.expanduser().resolve()
    if out==HERE or HERE in out.parents:p.error('La salida debe estar fuera de esta adenda.')
    if out.exists():p.error('La salida ya existe; elija una carpeta nueva.')
    work=out/'trabajo_figuras';shutil.copytree(HERE/'figuras/insumos',work)
    code=work/'codigo'
    shutil.copy2(HERE/'figuras/codigo_anterior/figuras_base_v9.py',code/'figuras_base_v9.py')
    manifest=json.loads((HERE/'figuras/procedencia/verificacion_figuras.json').read_text())
    for row in manifest['inputs']:
        assert hashlib.sha256((work/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    if a.prepare_only:
        print('Preparados los ocho insumos, sin renderizar:',out)
        return
    # Importación diferida: --help y --prepare-only no requieren bibliotecas científicas.
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    os.chdir(work);sys.path.insert(0,str(code))
    from estilo_figuras import apply_figure_style
    apply_figure_style()
    def namespace(name):
        path=code/name;g={'__file__':str(path)}
        exec(compile(path.read_text(),str(path),'exec'),g)
        return g
    def capture(ns,function):
        captured=[];ns['_save']=lambda fig,out:captured.append(fig)
        ns[function](out='no_output');assert len(captured)==1
        return captured[0]
    def numerical(fig):
        def coords(x):return np.asarray(x,dtype=float).tolist()
        return [{'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'xticks':coords(ax.get_xticks()),'yticks':coords(ax.get_yticks()),
                 'lines':[{'x':coords(z.get_xdata()),'y':coords(z.get_ydata()),'color':str(z.get_color()),'marker':str(z.get_marker()),'ls':str(z.get_linestyle())} for z in ax.lines],
                 'patches':[{'xy':coords(z.get_path().vertices),'transform':coords(z.get_patch_transform().get_matrix()),'color':coords(z.get_facecolor())} for z in ax.patches],
                 'collections':[{'offsets':coords(z.get_offsets()),'paths':[coords(v.vertices) for v in z.get_paths()]} for z in ax.collections]} for ax in fig.axes]
    old=namespace('figuras_base_v9.py');new=namespace('figuras_base_v9_DP.py')
    target=out/'figuras';target.mkdir()
    report=[]
    for name,fun in [('S1','figS1b'),('S2','figS2b')]:
        baseline=capture(old,fun);edited=capture(new,fun)
        assert numerical(baseline)==numerical(edited),'Los rótulos alteraron artistas numéricos'
        baseline.canvas.draw();edited.canvas.draw()
        box=baseline.get_tightbbox(baseline.canvas.get_renderer()).padded(.05)
        ref=next(x for x in manifest['verification'] if x['figure']==name)
        width,height=ref['output_inches']
        box=Bbox.from_bounds(box.x0+(box.width-width)/2,box.y0+(box.height-height)/2,width,height)
        overflow=[]
        for ax in edited.axes:
            for obj in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]:
                if obj.get_text():
                    b=obj.get_window_extent(edited.canvas.get_renderer()).transformed(edited.dpi_scale_trans.inverted())
                    if b.x0<box.x0-.01 or b.x1>box.x1+.01 or b.y0<box.y0-.01 or b.y1>box.y1+.01:
                        overflow.append(obj.get_text())
        assert not overflow,overflow
        for ext in ('pdf','png'):edited.savefig(target/f'Figura_{name}.{ext}',dpi=300,bbox_inches=box,pad_inches=0)
        report.append({'figura':name,'artistas_numericos_identicos':True,'output_inches':[width,height],
                       'texto_fuera_del_area':overflow,
                       'png_sha256':hashlib.sha256((target/f'Figura_{name}.png').read_bytes()).hexdigest(),
                       'png_identico_a_referencia':(target/f'Figura_{name}.png').read_bytes()==(HERE/f'figuras/referencia/Figura_{name}.png').read_bytes()})
        plt.close(baseline);plt.close(edited)
    (out/'verificacion_figuras_nueva.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
       'matplotlib':matplotlib.__version__,'verificacion':report,'nota':'PDF puede variar en metadatos de fecha; fuentes y versiones pueden cambiar bytes/píxeles sin cambiar valores.'},ensure_ascii=False,indent=2)+'\n')
    print('Figuras generadas sin reanalizar datos:',target)


if __name__=='__main__':main()
