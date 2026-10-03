"""Compara archivos de una ejecución con los resultados de referencia.

Regla: primero igualdad exacta de bytes. Si difiere y el archivo es CSV/CSV.GZ/JSON, se comparan
estructura (columnas, filas, claves) y valores: texto igual exacto; números con
|a-b| <= atol + rtol*|b| (atol=1e-12, rtol=1e-9 por omisión: margen de redondeo de punto flotante
de doble precisión acumulado en sumas y álgebra lineal; no se amplía para ocultar diferencias).
Uso: python comparar_resultados.py --referencia DIR --ejecucion DIR [--archivos LISTA] --salida informe.json
"""
import argparse, gzip, hashlib, io, json, math, sys
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _num(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None

def comparar_json(a, b, rtol, atol, ruta='', out=None):
    out = out if out is not None else {'max_abs': 0.0, 'max_rel': 0.0, 'n_num': 0, 'fallas': []}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): out['fallas'].append(f'{ruta}: claves distintas {sorted(set(a)^set(b))[:5]}')
        for k in set(a) & set(b): comparar_json(a[k], b[k], rtol, atol, f'{ruta}/{k}', out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out['fallas'].append(f'{ruta}: longitudes {len(a)} vs {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)): comparar_json(x, y, rtol, atol, f'{ruta}/{i}', out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if type(a) is not type(b) or a != b: out['fallas'].append(f'{ruta}: {str(b)[:60]!r} (ref) vs {str(a)[:60]!r}')
    else:
        _cmp_num(float(a), float(b), rtol, atol, ruta, out)
    return out

def _cmp_num(a, b, rtol, atol, ruta, out):
    out['n_num'] += 1
    if math.isnan(a) or math.isnan(b):
        if not (math.isnan(a) and math.isnan(b)):
            out['fallas'].append(f'{ruta}: {b!r} (ref) vs {a!r}; NaN unilateral')
        return
    if math.isinf(a) or math.isinf(b):
        if a != b: out['fallas'].append(f'{ruta}: {b} vs {a}')
        return
    d = abs(a - b); out['max_abs'] = max(out['max_abs'], d)
    if b != 0: out['max_rel'] = max(out['max_rel'], d / abs(b))
    if d > atol + rtol * abs(b) and len(out['fallas']) < 20: out['fallas'].append(f'{ruta}: {b!r} (ref) vs {a!r} (dif {d:.3g})')
    elif d > atol + rtol * abs(b): out['fallas'].append('...')

def leer_csv(p):
    import csv
    raw = Path(p).read_bytes()
    if str(p).endswith('.gz'): raw = gzip.decompress(raw)
    return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))

def comparar_csv(a, b, rtol, atol):
    A, B = leer_csv(a), leer_csv(b)
    out = {'max_abs': 0.0, 'max_rel': 0.0, 'n_num': 0, 'fallas': []}
    if len(A) != len(B): out['fallas'].append(f'filas {len(B)} (ref) vs {len(A)}'); return out
    for i, (ra, rb) in enumerate(zip(A, B)):
        if len(ra) != len(rb): out['fallas'].append(f'fila {i}: columnas {len(rb)} vs {len(ra)}'); continue
        for j, (x, y) in enumerate(zip(ra, rb)):
            if x == y: continue
            if i == 0:
                out['fallas'].append(f'encabezado col {j}: {y!r} (ref) vs {x!r}')
                continue
            xa, yb = _num(x), _num(y)
            if xa is None or yb is None: out['fallas'].append(f'fila {i} col {j}: {y!r} (ref) vs {x!r}')
            else: _cmp_num(xa, yb, rtol, atol, f'fila {i} col {j}', out)
    return out

def comparar(ref, run, rels, rtol=1e-9, atol=1e-12):
    filas = []
    for rel in rels:
        r, e = Path(ref) / rel, Path(run) / rel
        f = {'archivo': rel}
        if not r.is_file(): f.update(estado='sin_referencia')
        elif not e.is_file(): f.update(estado='no_generado')
        elif sha(r) == sha(e): f.update(estado='identico_bytes')
        elif str(rel).endswith('.gz') and gzip.decompress(r.read_bytes()) == gzip.decompress(e.read_bytes()):
            f.update(estado='identico_contenido', nota='solo difiere el encabezado gzip (fecha de compresión)')
        else:
            s = str(rel)
            if s.endswith(('.csv', '.csv.gz')): d = comparar_csv(e, r, rtol, atol)
            elif s.endswith('.json'): d = comparar_json(json.loads(e.read_text()), json.loads(r.read_text()), rtol, atol)
            else: d = None
            if d is None: f.update(estado='difiere_bytes_sin_comparador')
            else:
                f.update(estado='coincide_tolerancia' if not d['fallas'] else 'difiere', max_abs=d['max_abs'], max_rel=d['max_rel'], n_num=d['n_num'], fallas=d['fallas'][:10])
        filas.append(f)
    return filas

def resumen(filas):
    from collections import Counter
    return dict(Counter(f['estado'] for f in filas))

if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--referencia', required=True); a.add_argument('--ejecucion', required=True)
    a.add_argument('--archivos', help='archivo de texto con rutas relativas (una por línea); por omisión, todos los de la ejecución')
    a.add_argument('--salida', required=True); a.add_argument('--rtol', type=float, default=1e-9); a.add_argument('--atol', type=float, default=1e-12)
    x = a.parse_args()
    rels = Path(x.archivos).read_text().split() if x.archivos else sorted(str(p.relative_to(x.ejecucion)) for p in Path(x.ejecucion).rglob('*') if p.is_file())
    filas = comparar(x.referencia, x.ejecucion, rels, x.rtol, x.atol)
    Path(x.salida).write_text(json.dumps(filas, indent=1, ensure_ascii=False))
    print(resumen(filas))
