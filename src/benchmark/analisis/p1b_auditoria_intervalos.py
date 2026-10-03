"""P1b (v6). Alcance de los intervalos de predicción del cáncer de mama.
(1) Auditoría de la comprobación «0 de 190» de v5: composición por serie y horizonte, años de desenlace frente a los años de la
    calibración entre series y presencia de C50 o de agregados que lo contienen en el conjunto de calibración.
(2) Comprobación con la misma variable y la misma canalización de pronóstico que la aplicación (TEE y conteos de mama,
    pipeline_tsfm.py/pipeline_h48.py) y separación temporal: umbrales calculados con errores de conteos del corpus SIN series de
    cáncer de mama ni de «todas las causas», solo de años anteriores al año de prueba; prueba con los errores de mama del origen
    2016-12 (años 1-3 → 2017, 2018, 2019) y del origen 2015-12 a 48 meses (año 4 → 2019).
      año b (1-3): calibración corpus origen 2014-12 (año 2014+b) y 2015-12 (año 2015+b) → prueba mama 2016-12 (año 2016+b).
      año 4: calibración corpus h48 origen 2014-12 (año 2018) → prueba mama h48 2015-12 (año 2019).
(3) Sensibilidad de los intervalos 2020-2023 del texto al excluir C50 y «todas las causas» de la calibración.
Salidas: resultados/p1b_*.csv, p1b_resumen.json"""
import numpy as np, pandas as pd, json
ENS6 = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']; ENS = 'Ensamble (mediana)'
C = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(float)
BINS = [0, 10, 25, 100, 300, 1000, 1e7]; LAB = ['0-10', '10-25', '25-100', '100-300', '300-1000', '>1000']
C['vol'] = pd.cut(C.mean_pre, BINS, labels=LAB).astype(str); C['c50'] = C.grp.isin(['C. mama', 'Todas las causas'])
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from estratos import vol_corpus
C['vol14'] = vol_corpus(C, '2012-2014')  # v7: estratos de la comprobación separada (prueba 2017-2019) con años previos a toda prueba
def cq(e, a=0.9):
    e = np.sort(np.abs(np.asarray(e))); n = len(e); r = int(np.ceil((n + 1) * a)); return np.inf if r > n else e[r - 1]
M = pd.read_csv('resultados/benchmark/metricas_por_serie.csv.gz'); M = M[M.model == ENS]
L = pd.concat([M.assign(b=b, e=M[f'logerr_y{b}'])[['sid', 'origin', 'b', 'e']] for b in (1, 2, 3)])
for o, n in [('2014-12', 36), ('2015-12', 48)]:
    F = dict(np.load(f'pronosticos/publicados/fc_h48_{o}.npz')); q = np.median(np.stack([F[m] for m in ENS6]), 0); pt = q[..., 1]
    L = pd.concat([L, pd.DataFrame({'sid': C.index, 'origin': o, 'b': 4, 'e': np.log((Y[:, n + 36:n + 48].sum(1) + 1) / (pt[:, 36:48].sum(1) + 1))})])
L = L.merge(C[['vol', 'vol14', 'c50', 'grp', 'level']], left_on='sid', right_index=True); L['anio'] = L.origin.str[:4].astype(int) + L.b
S = pd.read_csv('datos/series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
def vol_of(tg, hasta='2019'):
    col = {'asmr': 'deaths', 'deaths_adj': 'deaths'}.get(tg, tg.replace('asmr_', 'deaths_'))
    return str(pd.cut([S.loc['2012':hasta, col].mean()], BINS, labels=LAB)[0])
bb = pd.read_csv('entradas/tsfm_backtest_bloques.csv'); b4 = pd.read_csv('entradas/tsfm_backtest_bloques_h48.csv')
bb = bb[bb.model == ENS]; b4 = b4[b4.model == ENS]
# ---------- (1) auditoría de los 190 ----------
Z = pd.concat([bb.assign(fuente='36 meses (9 orígenes)'), b4[b4.block == 4].assign(fuente='48 meses (5 orígenes)')])
Z['mes_ini'] = pd.to_datetime(Z.origin) + pd.DateOffset(months=1); Z['ini'] = [m + pd.DateOffset(months=12 * (b - 1)) for m, b in zip(Z.mes_ini, Z.block)]
Z['anios_desenlace'] = [f"{i.strftime('%Y-%m')} a {(i + pd.DateOffset(months=11)).strftime('%Y-%m')}" for i in Z.ini]
comp = Z.groupby(['fuente', 'block']).agg(series=('target', 'nunique'), origenes=('origin', 'nunique'), errores=('logerr', 'size'),
                                          desde=('ini', 'min'), hasta=('ini', 'max')).reset_index()
comp['hasta'] = comp.hasta + pd.DateOffset(months=11); comp['desde'] = comp.desde.dt.strftime('%Y-%m'); comp['hasta'] = comp.hasta.dt.strftime('%Y-%m')
comp.to_csv('resultados/p1b_composicion_190.csv', index=False)
cal_years = sorted(L[L.b <= 4].anio.unique().tolist())
pool = L.groupby(['b', 'vol']).agg(n=('e', 'size'), n_c50_o_todas=('c50', 'sum')).reset_index()
pool.to_csv('resultados/p1b_pool_calibracion.csv', index=False)
# ---------- (2) comprobación separada ----------
Lx = L[~L.c50]
rows = []
for b in (1, 2, 3, 4):
    if b < 4:
        cal = Lx[(Lx.b == b) & Lx.origin.isin(['2014-12', '2015-12'])]; tst = bb[(bb.origin == '2016-12') & (bb.block == b)]
    else:
        cal = Lx[(Lx.b == 4) & (Lx.origin == '2014-12')]; tst = b4[(b4.origin == '2015-12') & (b4.block == 4)]
    for _, r in tst.iterrows():
        v = vol_of(r.target, '2014'); ce = cal[cal.vol14 == v].e
        q = cq(ce); rows.append(dict(bloque=b, target=r.target, variable='tasa estandarizada' if r.target.startswith('asmr') else 'defunciones', anio_prueba=int(r.origin[:4]) + b, anios_calibracion=sorted(cal.anio.unique().tolist()),
                                     vol=v, n_cal=len(ce), q=q, error=r.logerr, fuera=abs(r.logerr) > q, razon_error_umbral=abs(r.logerr) / q))
T = pd.DataFrame(rows); T.to_csv('resultados/p1b_comprobacion_separada.csv', index=False)
# ---------- (3) sensibilidad de los intervalos del texto ----------
EX = pd.concat([pd.read_csv('entradas/tsfm_exceso_anual.csv'), pd.read_csv('entradas/tsfm_exceso_anual_subgrupos.csv')]); EX = EX[EX.model == ENS]
QT = {}
for excl, d in [('todas', L), ('sin_C50_ni_todas', Lx)]:
    for (b, v), g in d.groupby(['b', 'vol']): QT[(excl, b, v)] = cq(g.e.values)
sens = []
for _, r in EX[EX.year <= 2023].iterrows():
    b = r.year - 2019; v = vol_of(r.target); lr = np.log(r.observed / r.expected); rec = dict(target=r.target, year=int(r.year), pscore=r.pscore)
    for k in ('todas', 'sin_C50_ni_todas'):
        q = QT[(k, b, v)]; rec[f'lo_{k}'] = (np.exp(lr - q) - 1) * 100; rec[f'hi_{k}'] = (np.exp(lr + q) - 1) * 100
    sens.append(rec)
SE = pd.DataFrame(sens); SE.to_csv('resultados/p1b_sensibilidad_sin_C50.csv', index=False)
res = dict(n_190=int(len(Z)), composicion=comp.to_dict('records'), anios_calibracion_corpus=cal_years,
           breast_in_pool=dict(series_c50=int((C.grp == 'C. mama').sum()), series_todas=int((C.grp == 'Todas las causas').sum())),
           separada=dict(n=int(len(T)), n_fuera=int(T.fuera.sum()), n_tasas=int((T.variable == 'tasa estandarizada').sum()), n_conteos=int((T.variable == 'defunciones').sum()), estratos='2012-2014', razon_max=float(T.razon_error_umbral.max()), razon_mediana=float(T.razon_error_umbral.median())),
           sens_sig_sin_c50=int(((SE.lo_sin_C50_ni_todas > 0) | (SE.hi_sin_C50_ni_todas < 0)).sum()))
json.dump(res, open('resultados/p1b_resumen.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.set_option('display.width', 220)
print(comp.to_string(index=False)); print(pool[pool.vol.isin(['25-100', '100-300', '300-1000'])].to_string(index=False))
print(T[['bloque', 'target', 'anio_prueba', 'anios_calibracion', 'vol', 'n_cal', 'q', 'error', 'fuera']].round(3).to_string(index=False))
print(json.dumps(res['separada']), res['sens_sig_sin_c50'])
print(SE[SE.target == 'asmr'].round(1).to_string(index=False))
