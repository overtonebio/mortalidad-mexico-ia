"""Limpieza automática de tablas para publicación (material suplementario y tablas del texto).
limpiar(df, spec) aplica, columna por columna: renombrado, mapeo de etiquetas, número fijo de decimales,
separador de miles con espacio, signo menos tipográfico, formato de p (< 0.001) y '—' para faltantes.
verificar(df) devuelve la lista de problemas residuales (nombres internos, NaN, decimales largos, notación científica)."""
import re, numpy as np, pandas as pd

MINUS = '−'
MODEL_LABEL = {'Naive estacional': 'Ingenuo estacional', 'Log-lineal (tendencia+mes)': 'Log-lineal'}

def fnum(x, dec=1, sign=False, thousands=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or (isinstance(x, str) and x.strip().lower() in ('nan', '')):
        return '—'
    if isinstance(x, str): return x
    s = f'{x:+,.{dec}f}' if sign else f'{x:,.{dec}f}'
    s = s.replace(',', ' ' if thousands else '')
    return s.replace('-', MINUS)

def fp(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)): return '—'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'

def limpiar(df, spec, rename=None, labels=None, drop=None):
    """spec: {columna: ('num', dec) | ('sign', dec) | ('p',) | ('int',) | ('txt',)}"""
    d = df.copy()
    if drop: d = d.drop(columns=[c for c in drop if c in d.columns])
    if labels:
        for c, mp in labels.items():
            if c in d.columns: d[c] = d[c].map(lambda v: mp.get(v, v))
    for c, s in spec.items():
        if c not in d.columns: continue
        kind = s[0]
        if kind == 'num': d[c] = [fnum(v, s[1]) for v in d[c]]
        elif kind == 'sign': d[c] = [fnum(v, s[1], sign=True) for v in d[c]]
        elif kind == 'int': d[c] = [fnum(v, 0) for v in d[c]]
        elif kind == 'p': d[c] = [fp(v) for v in d[c]]
    d = d.fillna('—').astype(str).replace({'nan': '—', 'None': '—'})
    if rename: d = d.rename(columns=rename)
    return d

def verificar(df):
    prob = []
    for c in df.columns:
        if re.search(r'[a-z]+_[a-z0-9_]+', str(c)): prob.append(f'encabezado interno: {c}')
    for c in df.columns:
        for v in df[c].astype(str):
            if v in ('nan', 'None', 'NaN'): prob.append(f'{c}: faltante sin formato'); break
            if re.fullmatch(r'-?\d+\.\d{4,}', v): prob.append(f'{c}: decimales largos ({v})'); break
            if re.search(r'\de[+-]\d', v): prob.append(f'{c}: notación científica ({v})'); break
            if re.fullmatch(r'[a-z]+_[a-z0-9_]+', v): prob.append(f'{c}: etiqueta interna ({v})'); break
            if re.search(r'(?<![\w(])-\d', v): prob.append(f'{c}: guion en lugar de signo menos ({v})'); break
    return prob
