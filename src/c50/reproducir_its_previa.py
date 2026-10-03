"""Reproduce la ITS exploratoria previa en una copia aislada y compara con sus resultados guardados."""
import os, shutil, subprocess, sys, json, numpy as np, pandas as pd
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
PK = '../paquete_original/'; R = '../reproduccion/pk/'
shutil.rmtree(R, ignore_errors=True); os.makedirs(R + 'editorial')
shutil.copytree(PK + 'editorial/datos', R + 'editorial/datos'); shutil.copytree(PK + 'exploracion_previa', R + 'exploracion_previa')
r = subprocess.run([sys.executable, 'analisis_its.py'], cwd=R + 'exploracion_previa', capture_output=True, text=True)
open('../reproduccion/log_reproduccion.txt', 'w').write(r.stdout + r.stderr); assert r.returncode == 0
o, n = PK + 'exploracion_previa/', R + 'exploracion_previa/'
def flat(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[p + k] = v
    return out
a, b = flat(json.load(open(o + 'resultados_ITS.json'))), flat(json.load(open(n + 'resultados_ITS.json')))
res = {'json_claves': len(a), 'json_dif_max': max(abs(a[k] - b[k]) for k in a)}
for f in ['estimaciones_ITS.csv', 'serie_y_ajuste_principal.csv', 'resumen_anual.csv']:
    x, y = pd.read_csv(o + f), pd.read_csv(n + f); c = x.select_dtypes('number').columns; res[f] = float((x[c] - y[c]).abs().max().max())
json.dump(res, open('../reproduccion/comparacion_reproduccion.json', 'w'), indent=1); print(res)
assert max(v for k, v in res.items() if k != 'json_claves') < 1e-9
