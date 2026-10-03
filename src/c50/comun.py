"""Funciones comunes: lectura de microdatos C50, filtros, denominadores WPP y pesos OMS (idénticos a
prep_series_completo.py del paquete editorial)."""
from pathlib import Path
import os
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
# Resolve all legacy relative outputs within this portable package, never caller cwd.
os.chdir(ROOT / 'codigo')
RAW = '../paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz'
WPP = '../paquete_original/editorial/datos/wpp_mex_female_age5.csv'
SERIES = '../paquete_original/editorial/datos/series_mensuales_C50.csv'
WHO = pd.Series({20: 8220, 25: 7930, 30: 7610, 35: 7150, 40: 6590, 45: 6040, 50: 5370, 55: 4550, 60: 3720, 65: 2960,
                 70: 2210, 75: 1520, 80: 910, 85: 635}, dtype=float); WHO = WHO / WHO.sum()
IDX = pd.date_range('2012-01-01', '2024-12-01', freq='MS')
GRUPO_SITIO = {**{k: 'Unidad pública' for k in [1, 2, 3, 4, 5, 6, 7, 8, 13]}, 9: 'Unidad privada', 11: 'Hogar',
               10: 'Otro/NE', 12: 'Otro/NE', 99: 'Otro/NE'}
REG = {'Norte': [2, 3, 5, 8, 10, 19, 25, 26, 28], 'Centro-Occidente': [1, 6, 11, 14, 16, 18, 22, 24, 32],
       'Centro': [9, 13, 15, 17, 21, 29], 'Sur-Sureste': [4, 7, 12, 20, 23, 27, 30, 31]}
E2R = {e: r for r, l in REG.items() for e in l}

def leer_crudo():
    d = pd.read_csv(RAW, dtype=str)
    for c in d.columns: d[c] = d[c].str.strip()          # INEGI incluye tabuladores en claves geográficas
    return d

def filtrar(d, log=None):
    x = d.copy()
    for c in ['sexo', 'edad', 'mes_ocurr', 'anio_ocur', 'anio_regis', 'mes_regis', 'file_year', 'ent_resid', 'sitio_ocur']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['age'] = np.where((x.edad >= 4000) & (x.edad < 4998), x.edad - 4000, np.nan)
    pasos = [('registros C50 en archivos 2012-2024', np.ones(len(x), bool)), ('sexo = mujer', x.sexo == 2),
             ('edad válida ≥ 20 años', x.age >= 20), ('mes de ocurrencia válido', x.mes_ocurr.between(1, 12)),
             ('año de ocurrencia 2012-2024', x.anio_ocur.between(2012, 2024))]
    keep = np.ones(len(x), bool); rows = []
    for n, m in pasos:
        keep = keep & np.asarray(m); rows.append({'paso': n, 'n_restante': int(keep.sum())})
    w = x[keep].copy()
    w['date'] = pd.to_datetime(dict(year=w.anio_ocur, month=w.mes_ocurr, day=1))
    w['ag'] = (np.minimum(w.age, 85) // 5 * 5).astype(int)
    w['sitio'] = w.sitio_ocur.map(GRUPO_SITIO)
    w['residente'] = w.ent_resid.between(1, 32)
    return (w, pd.DataFrame(rows)) if log else w

def poblacion_mensual():
    p = pd.read_csv(WPP); p['ag'] = np.minimum(p.AgeGrpStart, 85)
    p = p[p.ag >= 20].groupby(['Time', 'ag']).PopFemale.sum().unstack() * 1000
    p.index = pd.to_datetime(p.index.astype(str) + '-07-01')
    t = IDX + pd.Timedelta(days=14)
    pm = p.reindex(p.index.union(t)).interpolate('time').reindex(t); pm.index = IDX
    return pm

def factor_2024(w):
    comp = (w[w.anio_ocur.between(2017, 2022)].assign(s=lambda x: x.anio_regis == x.anio_ocur).groupby('mes_ocurr').s.mean())
    fac = pd.Series(1.0, IDX); m = IDX.year == 2024; fac[m] = 1 / comp.reindex(IDX[m].month).values
    return fac

def tee(cnt, pm, ags=None):
    """cnt: DataFrame fecha x grupo de edad (conteos). Devuelve TEE por 100 000 mujeres-año."""
    ags = list(WHO.index) if ags is None else ags
    cnt = cnt.reindex(index=IDX, columns=WHO.index, fill_value=0)
    ww = WHO[ags] / WHO[ags].sum()
    return ((cnt[ags] / (pm[ags] / 12) * 1e5) * ww).sum(1)

def dk_cov(X, S, t, bread_inv, L=12, correccion='cluster'):
    """Covarianza Driscoll-Kraay: puntuaciones S (n x k) sumadas por periodo t (entero, meses calendario), núcleo de
    Bartlett sobre distancias calendario (respeta huecos), sándwich bread_inv @ meat @ bread_inv.
    correccion='cluster': G/(G-1) con G = número de periodos. Esta convención
    explícita conserva v10; no equivale a la opción homónima de statsmodels,
    que añade (nobs-1)/(nobs-k). Las ITS univariadas usan además n/(n-k)."""
    S = np.asarray(S); t = np.asarray(t).astype(int)
    per = np.unique(t); pos = {p: i for i, p in enumerate(per)}
    H = np.zeros((len(per), S.shape[1])); np.add.at(H, np.array([pos[v] for v in t]), S)
    meat = H.T @ H
    for lag in range(1, L + 1):
        a = [i for i, p in enumerate(per) if p - lag in pos]
        if not a: continue
        b = [pos[per[i] - lag] for i in a]; cr = H[a].T @ H[b]; meat += (1 - lag / (L + 1)) * (cr + cr.T)
    G = len(per); c = G / (G - 1) if correccion == 'cluster' else 1.0
    return c * bread_inv @ meat @ bread_inv

def diseno_its(fechas, corte, extra=None):
    f = pd.DatetimeIndex(fechas); mi = (f.year - 2012) * 12 + f.month - 1; c = (corte.year - 2012) * 12 + corte.month - 1
    X = pd.DataFrame({'const': 1.0, 'tiempo': mi / 12, 'post': (mi >= c).astype(float), 'tiempo_post': np.maximum(mi - c, 0) / 12})
    for m in range(2, 13): X[f'mes_{m}'] = (f.month == m).astype(float)
    return X, np.asarray(mi)

def contraste(b, C, v, z=1.959963984540054):
    v = np.asarray(v, float); e = float(v @ b); se = float(np.sqrt(v @ C @ v))
    from scipy import stats
    return {'log': e, 'se': se, 'pct': 100 * np.expm1(e), 'ic_inf': 100 * np.expm1(e - z * se), 'ic_sup': 100 * np.expm1(e + z * se),
            'p': float(2 * stats.norm.sf(abs(e / se)))}
