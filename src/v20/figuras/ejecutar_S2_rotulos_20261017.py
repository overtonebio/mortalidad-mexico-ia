"""Figura S2 con rótulo «C50» → «cáncer de mama» (17-oct-2026).
Sigue ejecutar_figuras.py de la adenda v15: (1) reproduce la Figura S2 de v19 (figuras_base_v9_DP.py, caja centrada
sobre la figura v13 con output_inches del manifiesto) y exige que sea idéntica píxel a píxel a figuras/referencia/Figura_S2.png;
(2) exige artistas numéricos idénticos entre v19 y la versión con rótulos; (3) exporta la versión nueva con la misma caja,
ampliada solo hacia arriba lo necesario para el título de dos líneas del panel c.
Uso: python ejecutar_S2_rotulos_20261017.py <adenda_v15> <salida>"""
import json, os, sys, shutil, hashlib
from pathlib import Path
A = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]).resolve(); HERE = Path(__file__).resolve().parent
work = out / 'trabajo'; shutil.copytree(A / 'figuras/insumos', work); code = work / 'codigo'
shutil.copy2(A / 'figuras/codigo_anterior/figuras_base_v9.py', code / 'figuras_base_v9.py')
shutil.copy2(HERE / 'figuras_base_v9_DP_rotulos_20261017.py', code / 'figuras_base_v9_DP_rotulos_20261017.py')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
os.chdir(work); sys.path.insert(0, str(code))
from estilo_figuras import apply_figure_style; apply_figure_style()
def ns(name):
    p = code / name; g = {'__file__': str(p)}; exec(compile(p.read_text(), str(p), 'exec'), g); return g
def capture(n, fun):
    c = []; n['_save'] = lambda fig, o: c.append(fig); n[fun](out='x'); assert len(c) == 1; return c[0]
def numerical(fig):
    co = lambda x: np.asarray(x, dtype=float).tolist()
    return [{'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()), 'xticks': co(ax.get_xticks()), 'yticks': co(ax.get_yticks()),
             'lines': [{'x': co(z.get_xdata()), 'y': co(z.get_ydata()), 'color': str(z.get_color()), 'marker': str(z.get_marker()), 'ls': str(z.get_linestyle())} for z in ax.lines],
             'patches': [{'xy': co(z.get_path().vertices), 'transform': co(z.get_patch_transform().get_matrix()), 'color': co(z.get_facecolor())} for z in ax.patches],
             'collections': [{'offsets': co(z.get_offsets()), 'paths': [co(v.vertices) for v in z.get_paths()]} for z in ax.collections]} for ax in fig.axes]
v13 = capture(ns('figuras_base_v9.py'), 'figS2b'); v19 = capture(ns('figuras_base_v9_DP.py'), 'figS2b'); nuevo = capture(ns('figuras_base_v9_DP_rotulos_20261017.py'), 'figS2b')
assert numerical(v13) == numerical(v19) == numerical(nuevo), 'cambió un artista numérico'
for f in (v13, v19, nuevo): f.canvas.draw()
man = json.loads((A / 'figuras/procedencia/verificacion_figuras.json').read_text())
w, h = next(x for x in man['verification'] if x['figure'] == 'S2')['output_inches']
b = v13.get_tightbbox(v13.canvas.get_renderer()).padded(.05)
caja_ref = Bbox.from_bounds(b.x0 + (b.width - w) / 2, b.y0 + (b.height - h) / 2, w, h)
(out / 'control').mkdir(parents=True); v19.savefig(out / 'control/Figura_S2_v19_regenerada.png', dpi=300, bbox_inches=caja_ref, pad_inches=0)
control_ok = (out / 'control/Figura_S2_v19_regenerada.png').read_bytes() == (A / 'figuras/referencia/Figura_S2.png').read_bytes()
assert control_ok, 'el entorno no reproduce la Figura S2 de v19'
t = nuevo.get_tightbbox(nuevo.canvas.get_renderer())
caja = Bbox([[caja_ref.x0, caja_ref.y0], [caja_ref.x1, max(caja_ref.y1, t.y1 + .05)]])
r = nuevo.canvas.get_renderer(); fuera = []
for ax in nuevo.axes:
    for o in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
        if o.get_text():
            e = o.get_window_extent(r).transformed(nuevo.dpi_scale_trans.inverted())
            if e.x0 < caja.x0 - .01 or e.x1 > caja.x1 + .01 or e.y0 < caja.y0 - .01 or e.y1 > caja.y1 + .01: fuera.append(o.get_text())
assert not fuera, fuera
(out / 'figuras').mkdir()
for ext in ('pdf', 'png'): nuevo.savefig(out / 'figuras' / f'Figura_S2.{ext}', dpi=300, bbox_inches=caja, pad_inches=0)
print(json.dumps({'control_v19_identico_a_referencia': control_ok, 'artistas_numericos_identicos': True,
  'output_inches_v19': [w, h], 'output_inches_nuevo': [caja.width, caja.height], 'ampliacion_superior_in': caja.height - h, 'texto_fuera': fuera}))
