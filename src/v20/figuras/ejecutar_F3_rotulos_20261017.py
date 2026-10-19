"""Figura 3 con rótulo del panel c «Cánceres» → «Tipo de cáncer» (17-oct-2026).
Reproduce el flujo de exportar_figuras_v13.py (fig2b con numeros_repro.json y atlas_repro.csv.gz) con el código
original y con la copia de rótulos; exige artistas numéricos idénticos y guarda PDF y PNG (300 ppp) con la misma
configuración de _save (bbox_inches='tight', pad_inches=0.05).
Uso (cwd = raíz de trabajo del benchmark): python ejecutar_F3_rotulos_20261017.py <dir_codigo> <salida>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]).resolve(); out.mkdir(parents=True)
sys.path.insert(0, str(C))
from estilo_figuras import apply_figure_style; apply_figure_style()
N = json.loads(Path('resultados/benchmark/numeros_repro.json').read_text()); PV = N['plac_vol']
plac_m = pd.DataFrame({m: {'semiancho_%': PV[m]['semiancho_mediana'], 'FPR_%': PV[m]['FPR']} for m in PV}).T
nat = pd.read_csv('resultados/benchmark/atlas_repro.csv.gz'); nat = nat[nat.level == 'Nacional']
def fig(name):
    g = {'__file__': str(C / name)}; exec((C / name).read_text(), g); cap = []
    g['_save'] = lambda f, o: cap.append(f); g['fig2b'](plac_m, nat, out='x'); return cap[0]
def numerical(f):
    co = lambda x: np.asarray(x, dtype=float).tolist()
    return [{'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()), 'xticks': co(ax.get_xticks()), 'yticks': co(ax.get_yticks()),
             'lines': [{'x': co(z.get_xdata()), 'y': co(z.get_ydata()), 'color': str(z.get_color()), 'marker': str(z.get_marker())} for z in ax.lines],
             'patches': [co(z.get_path().vertices) for z in ax.patches],
             'collections': [{'offsets': co(z.get_offsets()), 'paths': [co(v.vertices) for v in z.get_paths()]} for z in ax.collections]} for ax in f.axes]
f0 = fig('figuras_base_v9.py'); f1 = fig('figuras_base_v9_rotulos_20261017.py')
iguales = numerical(f0) == numerical(f1); assert iguales
t0 = f0.axes[2].get_title(); t1 = f1.axes[2].get_title()
f0.savefig(out / 'control_Figura_3_original.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
for e in ('pdf', 'png'): f1.savefig(out / f'Figura_3.{e}', dpi=300, bbox_inches='tight', pad_inches=0.05)
json.dump({'artistas_numericos_identicos': iguales, 'titulo_c_antes': t0, 'titulo_c_despues': t1}, open(out / 'resultado.json', 'w'), ensure_ascii=False, indent=1)
print(t0); print(t1)
