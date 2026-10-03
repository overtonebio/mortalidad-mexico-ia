"""Ampliaciones de la ITS exploratoria (plan 30-sep-2026, sección 4, y adenda 1). Salidas en ../resultados y ../tablas."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from comun import *
CUT = pd.Timestamp('2020-03-01'); R = {}; filas = []
s = pd.read_csv(SERIES, parse_dates=['fecha']); s23 = s[s.fecha < '2024-01-01'].reset_index(drop=True)

def ols_its(y, fechas, corte=CUT, L=12):
    X, t = diseno_its(fechas, corte); m = sm.OLS(np.log(y), X).fit()
    B = np.linalg.inv(X.T.values @ X.values); n, k = X.shape
    C = dk_cov(X.values, X.values * m.resid.values[:, None], t, B, L, correccion=None) * n / (n - k)
    return m, X, C, t

def efectos(b, C, cols, pref=''):
    ix = {c: i for i, c in enumerate(cols)}; out = {}
    for nom, terms in [('pendiente_pre', ['tiempo']), ('pendiente_post', ['tiempo', 'tiempo_post']),
                       ('cambio_pendiente', ['tiempo_post']), ('salto_nivel', ['post'])]:
        v = np.zeros(len(cols)); v[[ix[pref + t] for t in terms]] = 1; out[nom] = contraste(b, C, v)
    return out

# 0. Control: la covarianza propia reproduce la ITS previa (serie contigua)
m, X, C, t = ols_its(s23.asmr.values, s23.fecha)
ref = m.get_robustcov_results(cov_type='HAC', maxlags=12, use_correction=True, use_t=False).cov_params()
assert np.allclose(ref, C, rtol=1e-8, atol=1e-12)
R['principal'] = efectos(m.params.values, C, list(X.columns))
# 1. Diagnósticos de residuos del modelo principal
r = m.resid.values
R['diagnostico_principal'] = {'ljung_box': acorr_ljungbox(r, lags=[6, 12, 24], return_df=True).round(4).to_dict('index'),
    'durbin_watson': float(durbin_watson(r)), 'acf_1_12': [float(v) for v in sm.tsa.acf(r, nlags=12)[1:]],
    'shapiro_p': float(stats.shapiro(r).pvalue), 'residuo_sd_log': float(np.std(r, ddof=X.shape[1])),
    'residuo_max_abs': float(np.abs(r).max()), 'fecha_residuo_max': str(s23.fecha[np.argmax(np.abs(r))].date()),
    'residuo_medio_pre': float(r[s23.fecha < CUT].mean()), 'residuo_medio_post': float(r[s23.fecha >= CUT].mean()),
    'residuo_medio_anual': {int(y): float(v) for y, v in pd.Series(r).groupby(s23.fecha.dt.year).mean().items()}}
pd.DataFrame({'fecha': s23.fecha, 'residuo_log': r, 'ajustado_log': m.fittedvalues}).to_csv('../resultados/residuos_ITS_principal.csv', index=False)

# 2. Modelo conjunto apilado por edad con Driscoll-Kraay
G = ['20-39', '40-59', '60+']; bloques = []; ys = []; ts = []
for g in G:
    Xg, tg = diseno_its(s23.fecha, CUT); Xg.columns = [f'{g}:{c}' for c in Xg.columns]; bloques.append(Xg); ys.append(np.log(s23[f'asmr_{g}'].values)); ts.append(tg)
XS = pd.concat(bloques, axis=0, ignore_index=True).fillna(0.0); yS = np.concatenate(ys); tS = np.concatenate(ts)
mS = sm.OLS(yS, XS).fit(); BS = np.linalg.inv(XS.T.values @ XS.values)
CS = dk_cov(XS.values, XS.values * mS.resid.values[:, None], tS, BS, 12, 'cluster')
chk = mS.get_robustcov_results(cov_type='hac-groupsum', time=tS, maxlags=12, use_correction=False).cov_params()
R['DK_control_statsmodels_max_rel'] = float(np.max(np.abs(CS * (143 / 144) - chk)) / np.max(np.abs(chk)))
cols = list(XS.columns); ix = {c: i for i, c in enumerate(cols)}
R['edad_conjunto'] = {g: efectos(mS.params.values, CS, cols, f'{g}:') for g in G}
def wald(par):
    A = np.zeros((2, len(cols)))
    for i, (g1, g2) in enumerate([(G[0], G[1]), (G[1], G[2])]): A[i, ix[f'{g1}:{par}']] = 1; A[i, ix[f'{g2}:{par}']] = -1
    b = A @ mS.params.values; W = float(b @ np.linalg.solve(A @ CS @ A.T, b)); return {'chi2': W, 'gl': 2, 'p': float(stats.chi2.sf(W, 2))}
R['heterogeneidad_edad'] = {p: wald(p) for p in ['tiempo_post', 'post', 'tiempo']}
pares = [(G[0], G[1]), (G[0], G[2]), (G[1], G[2])]; pp = []
for g1, g2 in pares:
    v = np.zeros(len(cols)); v[ix[f'{g1}:tiempo_post']] = 1; v[ix[f'{g2}:tiempo_post']] = -1; e = contraste(mS.params.values, CS, v); pp.append(e['p'])
    R.setdefault('pares_cambio_pendiente', {})[f'{g1} vs {g2}'] = e
for (g1, g2), ph in zip(pares, multipletests(pp, method='holm')[1]): R['pares_cambio_pendiente'][f'{g1} vs {g2}']['p_holm'] = float(ph)
pp_nivel = []
for g1, g2 in pares:
    v = np.zeros(len(cols)); v[ix[f'{g1}:post']] = 1; v[ix[f'{g2}:post']] = -1
    e = contraste(mS.params.values, CS, v); pp_nivel.append(e['p'])
    R.setdefault('pares_cambio_nivel', {})[f'{g1} vs {g2}'] = e
for (g1, g2), ph in zip(pares, multipletests(pp_nivel, method='holm')[1]):
    R['pares_cambio_nivel'][f'{g1} vs {g2}']['p_holm'] = float(ph)
R['covarianza_edades'] = {'metodo': 'Scores apilados sumados por mes, Bartlett12',
    'correccion': 'T/(T-1)', 'factor': 144/143, 'T': 144, 'k_por_ecuacion': 15,
    'eleccion': 'Se conserva la convención explícita de Claude v10; no es la opción cluster completa de statsmodels.',
    'sensibilidad_correccion_n_nk': {'factor': 144/129,
        'chi2_cambio_pendiente': R['heterogeneidad_edad']['tiempo_post']['chi2'] * 129/143,
        'p_cambio_pendiente': float(stats.chi2.sf(R['heterogeneidad_edad']['tiempo_post']['chi2'] * 129/143, 2))},
    'interpretacion': 'Heterogeneidad de cambio de pendiente; no prueba diferencias de todas las pendientes posteriores.'}

# 3. Solo residentes en las 32 entidades
w = pd.read_pickle('../resultados/c50_analitico.pkl'); pm = poblacion_mensual(); fac = factor_2024(w)
cr = w[w.residente].groupby(['date', 'ag']).size().unstack(fill_value=0).reindex(IDX, fill_value=0).mul(fac, axis=0)
tee_res = tee(cr, pm); y = tee_res[IDX < '2024-01-01']
m3, X3, C3, _ = ols_its(y.values, y.index); R['solo_residentes'] = efectos(m3.params.values, C3, list(X3.columns))
R['solo_residentes_excluidos_2012_2023'] = int((~w.residente & (w.anio_ocur <= 2023)).sum())

# 4. 2023 inflado por registro tardío residual esperado
V = json.load(open('../resultados/validacion_datos.json'))['residual_2023']
for nom, q in [('media', V['proporcion_ge2_media']), ('maximo', V['proporcion_ge2_max'])]:
    yy = s23.asmr.values * np.where(s23.fecha.dt.year == 2023, 1 / (1 - q), 1.0)
    m4, X4, C4, _ = ols_its(yy, s23.fecha); R[f'registro_tardio_2023_{nom}'] = efectos(m4.params.values, C4, list(X4.columns))

# 5. Poisson por edad con covarianza robusta (nombre heredado cuasi-Poisson, adenda1)
c = w[w.anio_ocur <= 2023].groupby(['date', 'ag']).size().rename('d').reset_index()
grid = pd.MultiIndex.from_product([IDX[IDX < '2024-01-01'], WHO.index], names=['date', 'ag']).to_frame(index=False)
c = grid.merge(c, how='left').fillna({'d': 0}); c['py'] = [pm.loc[a, b] / 12 for a, b in zip(c.date, c.ag)]
Xc, tc = diseno_its(c.date, CUT); Xc = pd.concat([Xc, pd.get_dummies(c.ag, prefix='edad', drop_first=True).astype(float)], axis=1)
g = sm.GLM(c.d.values, Xc, family=sm.families.Poisson(), offset=np.log(c.py.values)).fit()
mu = g.fittedvalues.values; Bc = np.linalg.inv(Xc.T.values @ (Xc.values * mu[:, None]))
Cc = dk_cov(Xc.values, Xc.values * (c.d.values - mu)[:, None], tc, Bc, 12, 'cluster')
R['cuasi_poisson_edad'] = efectos(g.params.values, Cc, list(Xc.columns))
R['cuasi_poisson_dispersion'] = float(g.pearson_chi2 / g.df_resid)
Hc = Xc.T.values @ (Xc.values * mu[:,None])
scorec = Xc.T.values @ (c.d.values-mu)
assert g.converged and np.isfinite(g.params).all() and np.linalg.matrix_rank(Xc.values)==Xc.shape[1]
assert np.linalg.cond(Hc)<1e10 and np.max(np.abs(scorec))/c.d.sum()<1e-7 and np.linalg.eigvalsh(Cc).min()>-1e-10
R['metodo_poisson'] = {'nombre': 'Poisson con covarianza robusta de scores por mes (Bartlett12)',
    'clave_heredada': 'cuasi_poisson_edad', 'correccion': 'T/(T-1)',
    'exposicion': 'Población femenina WPP del quinquenio /12, mujeres-año',
    'n_celdas': int(len(c)), 'n_defunciones': int(c.d.sum()), 'n_edades': int(c.ag.nunique()),
    'convergencia': bool(g.converged), 'H_condicion':float(np.linalg.cond(Hc)), 'score_max_abs':float(np.max(np.abs(scorec))), 'dispersion_Pearson': R['cuasi_poisson_dispersion'],
    'limite': 'Términos temporales comunes a todas las edades; efecto ajustado distinto de la TEE directa. No se multiplica de nuevo por dispersión.'}
R['registro_2023_escenario'] = {'fracciones': V,
    'alcance': 'Escenario con fracciones históricas observadas hasta2024; no estimación validada de todo el faltante.',
    'supuestos': 'Cola histórica con seguimiento desigual; retrasos transportables; factor uniforme por edad y mes; incertidumbre del factor no propagada.'}

# tabla completa
def add(analisis, serie, E):
    for k, e in E.items(): filas.append({'analisis': analisis, 'serie': serie, 'parametro': k, 'estimacion_pct': e['pct'], 'ic95_inf': e['ic_inf'], 'ic95_sup': e['ic_sup'], 'p': e['p']})
add('principal (control)', 'Nacional', R['principal'])
for g_ in G: add('edad, modelo conjunto DK', g_, R['edad_conjunto'][g_])
add('solo residentes 1-32', 'Nacional', R['solo_residentes'])
add('2023 + registro tardío (media)', 'Nacional', R['registro_tardio_2023_media'])
add('2023 + registro tardío (máximo)', 'Nacional', R['registro_tardio_2023_maximo'])
add('cuasi-Poisson por edad', 'Nacional', R['cuasi_poisson_edad'])
pairs=[]
for family in ['cambio_pendiente','cambio_nivel']:
    for pair,entry in R['pares_'+family].items(): pairs.append({'familia':family,'par':pair,**entry})
pd.DataFrame(pairs).to_csv('../tablas/tablaS_contrastes_edad_Holm.csv',index=False)
pd.DataFrame(filas).to_csv('../tablas/tablaS_tendencia_ampliaciones.csv', index=False)
json.dump(R, open('../resultados/tendencia_ampliaciones.json', 'w'), indent=1, ensure_ascii=False)
print(pd.DataFrame(filas).round(3).to_string(index=False))
print(json.dumps({k: R[k] for k in ['heterogeneidad_edad', 'DK_control_statsmodels_max_rel', 'cuasi_poisson_dispersion', 'solo_residentes_excluidos_2012_2023']}, indent=0))
print({k: round(v['p_holm'], 4) for k, v in R['pares_cambio_pendiente'].items()})
d = R['diagnostico_principal']; print({k: d[k] for k in ['durbin_watson', 'shapiro_p', 'fecha_residuo_max', 'residuo_medio_anual']}, d['ljung_box'])
