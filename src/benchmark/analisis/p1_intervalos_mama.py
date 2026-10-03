"""P1 (v5; v7: placebo con estratos 2012-2014, aplicación 2020-2024 con estratos 2012-2019). Intervalos anuales del contrafactual de cáncer de mama (2020-2024).
(a) Diagnóstico de los intervalos por serie (9 orígenes trimestrales solapados, años 1-3; 5 orígenes, año 4):
    rango requerido ceil((n+1)·0.9), nivel nominal del máximo n/(n+1) bajo intercambiabilidad, solapamiento de ventanas
    y correlación entre errores de orígenes consecutivos.
(b) Calibración alternativa entre series (corpus de 2 870 series, errores anuales en log del ensamble; estratos de
    volumen de Mondrian). Años 1-3: 3 orígenes (2014-12, 2015-12, 2016-12); año 4: 2 orígenes h=48 (2014-12, 2015-12).
    Placebo independiente: años 1-3 calibración 2014+2015 → prueba 2016; año 4 calibración 2014 (año 4 = 2018)
    → prueba 2015 (año 4 = 2019; ventanas sin solapamiento). Variante: solo series nacionales (nacional y nacional×edad).
(c) Aplicación al punto del ensamble de la canalización de mama (tsfm_exceso_anual*.csv, v4; 2023 = h48).
    2024 (año 5) no tiene errores de validación: se informa solo como extrapolación descriptiva con el cuantil del año 4.
Supuesto: los errores relativos de las series de conteos del corpus son intercambiables con los de las tasas
estandarizadas de mama del mismo estrato de volumen (no verificable directamente; ver p1_resumen.json).
Salidas: resultados/p1_*.csv, p1_resumen.json"""
import numpy as np, pandas as pd, json
Q = np.array([0.1, 0.5, 0.9]); ENS6 = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']
ENS = 'Ensamble (mediana)'
C = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(float)
BINS = [0, 10, 25, 100, 300, 1000, 1e7]; LAB = ['0-10', '10-25', '25-100', '100-300', '300-1000', '>1000']
C['vol'] = pd.cut(C.mean_pre, BINS, labels=LAB).astype(str)
C['nac'] = C.level.isin(['Nacional', 'Nacional×edad'])
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from estratos import vol_corpus
C['vol14'] = vol_corpus(C, '2012-2014')  # v7: estratos del placebo (prueba 2016-2019) fijados con años previos a toda prueba
def cq(e, a=0.9):
    e = np.abs(np.asarray(e)); n = len(e); r = int(np.ceil((n + 1) * a))
    return np.inf if r > n else np.sort(e)[r - 1]
# ---------- (a) diagnóstico por serie ----------
bb = pd.read_csv('entradas/tsfm_backtest_bloques.csv'); b4 = pd.read_csv('entradas/tsfm_backtest_bloques_h48.csv')
diag = []
for tg in bb.target.unique():
    for b, src in [(1, bb), (2, bb), (3, bb), (4, b4)]:
        e = src[(src.target == tg) & (src.model == ENS) & (src.block == b)].sort_values('origin').logerr.values
        n = len(e); r = int(np.ceil((n + 1) * 0.9))
        diag.append(dict(target=tg, bloque=b, n_errores=n, rango_requerido=r, rango_usado=min(r, n),
                         nivel_nominal_max=n / (n + 1) * 100, solapamiento_consecutivo_meses=9,
                         autocorr_lag1=float(np.corrcoef(e[:-1], e[1:])[0, 1]) if n > 2 else np.nan,
                         sesgo_medio_pct=(np.exp(e.mean()) - 1) * 100, max_abs=np.abs(e).max()))
D = pd.DataFrame(diag); D.round(4).to_csv('resultados/p1_diagnostico_por_serie.csv', index=False)
# ---------- (b) errores entre series ----------
M = pd.read_csv('resultados/benchmark/metricas_por_serie.csv.gz')
M = M[M.model == ENS][['sid', 'origin', 'logerr_y1', 'logerr_y2', 'logerr_y3']]
E4 = []
for o, n in [('2014-12', 36), ('2015-12', 48)]:
    F = dict(np.load(f'pronosticos/publicados/fc_h48_{o}.npz')); q = np.median(np.stack([F[m] for m in ENS6]), 0); pt = q[..., 1]
    sl = slice(n + 36, n + 48); E4.append(pd.DataFrame({'sid': C.index, 'origin': o, 'logerr_y4': np.log((Y[:, sl].sum(1) + 1) / (pt[:, 36:48].sum(1) + 1))}))
E4 = pd.concat(E4)
L = pd.concat([M.melt(['sid', 'origin'], var_name='b', value_name='err'), E4.melt(['sid', 'origin'], var_name='b', value_name='err')])
L['bloque'] = L.b.str[-1].astype(int); L = L.merge(C[['vol', 'vol14', 'nac', 'level']], left_on='sid', right_index=True)
def calib(df, key):
    return df.groupby(key).err.apply(lambda e: cq(e.values)).rename('q').reset_index()
plac = []
for var, filt in [('Mondrian volumen (todas las series)', lambda d: d), ('Solo series nacionales, por volumen', lambda d: d[d.nac])]:
    d = filt(L).assign(vol=lambda x: x.vol14)  # placebo: estratos 2012-2014
    for b in (1, 2, 3, 4):
        cal_o, tst_o = (['2014-12', '2015-12'], '2016-12') if b < 4 else (['2014-12'], '2015-12')
        cal = d[(d.bloque == b) & d.origin.isin(cal_o)]; tst = d[(d.bloque == b) & (d.origin == tst_o)]
        qq = calib(cal, 'vol'); t = tst.merge(qq, on='vol')
        t['fp'] = t.err.abs() > t.q
        for v, g in t.groupby('vol'):
            plac.append(dict(variante=var, bloque=b, vol=v, n_cal=int((cal.vol == v).sum()), n_prueba=len(g), FPR=g.fp.mean() * 100,
                             sesgo_prueba_pct=(np.exp(g.err.mean()) - 1) * 100))
        plac.append(dict(variante=var, bloque=b, vol='todos', n_cal=len(cal), n_prueba=len(t), FPR=t.fp.mean() * 100,
                         sesgo_prueba_pct=(np.exp(t.err.mean()) - 1) * 100))
P = pd.DataFrame(plac); P.round(2).to_csv('resultados/p1_placebo_calibracion_entre_series.csv', index=False)
# cuantiles finales (todos los orígenes disponibles)
QT = {}
for var, filt in [('todas', lambda d: d), ('nacionales', lambda d: d[d.nac])]:
    d = filt(L)
    for (b, v), g in d.groupby(['bloque', 'vol']): QT[(var, b, v)] = (cq(g.err.values), len(g))
# ---------- (c) aplicación a mama ----------
S = pd.read_csv('datos/series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
def vol_of(tg):
    col = {'asmr': 'deaths', 'deaths_adj': 'deaths'}.get(tg, tg.replace('asmr_', 'deaths_'))
    m = S.loc['2012':'2019', col].mean(); return str(pd.cut([m], BINS, labels=LAB)[0]), m
EX = pd.concat([pd.read_csv('entradas/tsfm_exceso_anual.csv'), pd.read_csv('entradas/tsfm_exceso_anual_subgrupos.csv')])
EX = EX[EX.model == ENS].copy()
out = []
for _, r in EX.iterrows():
    v, mvol = vol_of(r.target); b = min(r.year - 2019, 4)
    rec = dict(target=r.target, year=int(r.year), observado=r.observed, esperado=r.expected, pscore=r.pscore,
               rango_serie_lo=r.p_lo, rango_serie_hi=r.p_hi, vol=v, defunciones_mes_2012_2019=mvol, bloque_cuantil=b,
               validado=bool(r.year <= 2023))
    lr = np.log(r.observed / r.expected)
    for var in ('todas', 'nacionales'):
        q, n = QT[(var, b, v)]
        rec[f'q_{var}'] = q; rec[f'n_cal_{var}'] = n
        rec[f'lo_{var}'] = (np.exp(lr - q) - 1) * 100; rec[f'hi_{var}'] = (np.exp(lr + q) - 1) * 100
    out.append(rec)
O = pd.DataFrame(out); O.round(4).to_csv('resultados/p1_intervalos_mama_entre_series.csv', index=False)
res = {'diag': D[D.target == 'asmr'].round(4).to_dict('records'),
       'placebo_ens': P.round(2).to_dict('records')}

# ---------- (d) conservadurismo: errores de validación de las 11 series de mama frente al cuantil entre series ----------
b4s = pd.read_csv('entradas/tsfm_backtest_bloques_h48.csv')
allb = pd.concat([bb, b4s[b4s.block == 4]]); allb = allb[allb.model == ENS].copy()
allb['vol'] = [vol_of(t)[0] for t in allb.target]
allb['q'] = [QT[('todas', b, v)][0] for b, v in zip(allb.block, allb.vol)]
res['backtest_mama_supera_cuantil_entre_series'] = dict(n=int(len(allb)), n_supera=int((allb.logerr.abs() > allb.q).sum()))
# ---------- (e) v8: sensibilidad con el modelo log-lineal (mismo procedimiento, errores del log-lineal) ----------
ML = pd.read_csv('resultados/benchmark/metricas_por_serie.csv.gz'); ML = ML[ML.model == 'Log-lineal'][['sid', 'origin', 'logerr_y1', 'logerr_y2', 'logerr_y3']]
E4L = []
for o, n in [('2014-12', 36), ('2015-12', 48)]:
    F = dict(np.load(f'pronosticos/publicados/fc_h48_{o}.npz')); pt = F['Log-lineal'][..., 1]
    E4L.append(pd.DataFrame({'sid': C.index, 'origin': o, 'logerr_y4': np.log((Y[:, n + 36:n + 48].sum(1) + 1) / (pt[:, 36:48].sum(1) + 1))}))
LL_ = pd.concat([ML.melt(['sid', 'origin'], var_name='b', value_name='err'), pd.concat(E4L).melt(['sid', 'origin'], var_name='b', value_name='err')])
LL_['bloque'] = LL_.b.str[-1].astype(int); LL_ = LL_.merge(C[['vol']], left_on='sid', right_index=True)
QL = {(b, v): cq(g.err.values) for (b, v), g in LL_.groupby(['bloque', 'vol'])}
EXL = pd.concat([pd.read_csv('entradas/tsfm_exceso_anual.csv'), pd.read_csv('entradas/tsfm_exceso_anual_subgrupos.csv')])
EXL = EXL[EXL.model == 'Log-lineal (tendencia+mes)']
outL = []
for _, r in EXL.iterrows():
    v, _m = vol_of(r.target); b = min(r.year - 2019, 4); lr = np.log(r.observed / r.expected); q = QL[(b, v)]
    outL.append(dict(target=r.target, year=int(r.year), pscore=r.pscore, lo=(np.exp(lr - q) - 1) * 100, hi=(np.exp(lr + q) - 1) * 100, validado=bool(r.year <= 2023)))
pd.DataFrame(outL).round(4).to_csv('resultados/p1_intervalos_mama_loglineal.csv', index=False)
# MASE de la validación específica de mama (TEE nacional, 9 orígenes trimestrales solapados; escala = error del ingenuo estacional en entrenamiento)
BM = pd.read_csv('entradas/tsfm_backtest_mensual.csv'); BM = BM[BM.target == 'asmr']
wide = BM.pivot(index='origin', columns='model', values='MASE')
res['mase_mama_tee_nacional'] = dict(media=BM.groupby('model').MASE.mean().round(4).to_dict(), n_origenes=int(BM.origin.nunique()),
    origenes_ensamble_menor_que_loglineal=int((wide['Ensamble (mediana)'] < wide['Log-lineal (tendencia+mes)']).sum()))
json.dump(res, open('resultados/p1_resumen.json', 'w'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250)
print(D[D.target == 'asmr'].round(3).to_string())
print(P[P.vol.isin(['todos', '300-1000', '100-300', '25-100'])].round(1).to_string())
print(O[['target', 'year', 'pscore', 'rango_serie_lo', 'rango_serie_hi', 'vol', 'q_todas', 'lo_todas', 'hi_todas', 'q_nacionales', 'n_cal_nacionales', 'lo_nacionales', 'hi_nacionales']].round(1).to_string())
