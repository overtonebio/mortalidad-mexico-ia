"""Lugar de defunción por C50 (plan 30-sep-2026, sección 3, y adendas 1-2). Binomial por celdas mes x edad x entidad,
Driscoll-Kraay (12 rezagos), estandarización marginal y simulación paramétrica (10 000 extracciones, semilla 20260930)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, time, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from comun import *
NSIM = int(os.environ.get('NSIM', 10000)); SEED = 20260930
w = pd.read_pickle('../resultados/c50_analitico.pkl')
w['sitio_alt'] = np.where(w.sitio_ocur == 99, 'No especificado', np.where(w.sitio_ocur.isin([10, 12]), 'Vía pública/otro', w.sitio))
w['region'] = w.ent_resid.map(E2R)
CATS = ['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE']
COMP = w[w.anio_ocur.between(2017, 2022)].assign(s=lambda x: x.anio_regis == x.anio_ocur).groupby(['sitio', 'mes_ocurr']).s.mean()

def celdas(d, geo='ent_resid', cat='sitio', pesos=False):
    d = d.copy(); d['wt'] = 1.0
    if pesos:
        m = d.anio_ocur == 2024; d.loc[m, 'wt'] = 1 / COMP.reindex(list(zip(d.loc[m, 'sitio'], d.loc[m, 'mes_ocurr']))).values
    t = d.pivot_table(index=['date', 'ag', geo], columns=cat, values='wt', aggfunc='sum', fill_value=0).reset_index()
    t['n'] = t[[c for c in t.columns if c not in ('date', 'ag', geo)]].sum(1); return t

def diseno(t, geo, corte, tipo='segmentado', ref_geo=None):
    f = pd.DatetimeIndex(t.date); mi = (f.year - 2012) * 12 + f.month - 1; c = (corte.year - 2012) * 12 + corte.month - 1
    X = pd.DataFrame({'const': np.ones(len(t))})
    if tipo == 'segmentado':
        X['tiempo'] = mi / 12; X['post'] = (mi >= c).astype(float); X['tiempo_post'] = np.maximum(mi - c, 0) / 12
    elif tipo == 'anual_post':
        X['tiempo'] = mi / 12
        for y in [2020, 2021, 2022, 2023]: X[f'p{y}'] = ((mi >= c) & (f.year == y)).astype(float)
    elif tipo == 'anios':
        for y in range(2013, 2025): X[f'a{y}'] = (f.year == y).astype(float)
    for m in range(2, 13): X[f'mes_{m}'] = (f.month == m).astype(float)
    for a in WHO.index[1:]: X[f'edad_{a}'] = (t.ag.values == a).astype(float)
    levs = sorted(t[geo].unique()) if ref_geo is None else ref_geo
    for g in levs[1:]: X[f'geo_{g}'] = (t[geo].values == g).astype(float)
    return X, np.asarray(mi), levs

def ajustar(t, y, geo, corte, tipo, L=12, ref_geo=None):
    X, mi, levs = diseno(t, geo, corte, tipo, ref_geo); X = X.loc[:, (X != 0).any()]
    k = t[y].values.astype(float); n = t.n.values.astype(float)
    diag = {'n_parametros': int(X.shape[1]), 'n_meses': int(len(np.unique(mi))),
            'rango_diseno': int(np.linalg.matrix_rank(X.values)), 'correccion_covarianza': 'T/(T-1)',
            'rezagos': L, 'separacion_estructural': [], 'inferencial': False}
    for variable in ['ag', geo]:
        audit = t.assign(eventos=k, resto=n-k).groupby(variable)[['eventos','resto']].sum()
        for level, row in audit.iterrows():
            if row.eventos == 0 or row.resto == 0:
                diag['separacion_estructural'].append({'variable': variable, 'nivel': int(level) if isinstance(level, (int, np.integer)) else str(level),
                    'eventos': float(row.eventos), 'resto': float(row.resto)})
    if diag['separacion_estructural'] or diag['rango_diseno'] != X.shape[1]:
        diag['estado'] = 'No estimable: separación estructural o diseño sin rango completo; sin OR/IC/p.'
        diag['intento'] = 'Validación previa al ajuste; no se publica una MLE sin existencia finita.'
        return None, X, None, levs, diag
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            g = sm.GLM(np.column_stack([k, n - k]), X, family=sm.families.Binomial()).fit(tol=1e-10, maxiter=100)
        diag['advertencias'] = [str(v.message) for v in caught]
    except Exception as exc:
        diag['estado'] = 'Error de ajuste; sin inferencia'; diag['error'] = str(exc)
        return None, X, None, levs, diag
    p = g.fittedvalues.values if g.fittedvalues.values.max() <= 1 else g.fittedvalues.values / n
    H = X.T.values @ (X.values * (n * p * (1 - p))[:, None])
    score = X.T.values @ (k-n*p)
    diag.update({'convergencia': bool(g.converged), 'H_condicion': float(np.linalg.cond(H)),
                 'score_max_abs': float(np.max(np.abs(score))), 'coeficientes_finitos': bool(np.isfinite(g.params).all())})
    if not (g.converged and diag['coeficientes_finitos'] and diag['H_condicion'] < 1e10 and diag['score_max_abs']/n.sum() < 1e-7):
        diag['estado'] = 'Ajuste no válido: convergencia, condición o score; sin OR/IC/p.'
        return g, X, None, levs, diag
    B = np.linalg.inv(H)
    C = dk_cov(X.values, X.values * (k - n * p)[:, None], mi, B, L, 'cluster')
    diag['cov_eigen_min'] = float(np.linalg.eigvalsh((C+C.T)/2).min())
    diag['inferencial'] = bool(np.isfinite(C).all() and diag['cov_eigen_min'] > -1e-10)
    diag['estado'] = 'Válido para inferencia exploratoria condicional al modelo' if diag['inferencial'] else 'Covarianza no válida; sin OR/IC/p.'
    observed = np.bincount(mi-mi.min(), weights=k)
    fitted = np.bincount(mi-mi.min(), weights=n*p)
    variance = np.bincount(mi-mi.min(), weights=n*p*(1-p))
    resid = (observed-fitted)/np.sqrt(variance)
    diag['residuos_mensuales'] = {'tipo': 'Pearson agregado por mes',
        'ljung_box': acorr_ljungbox(resid,lags=[6,12,24],return_df=True).to_dict('index'),
        'autocorrelacion_1': float(np.corrcoef(resid[:-1],resid[1:])[0,1]),
        'nota': 'Diagnóstico aproximado; HAC no garantiza forma temporal correcta.'}
    return g, X, C if diag['inferencial'] else None, levs, diag

def estandarizar(beta, cols, strata, geo, meses, corte, tipo, contrafactual=False, levs=None):
    """beta: (S x k). Devuelve (S x len(meses)) probabilidades estandarizadas sobre strata (ag, geo, peso)."""
    out = np.empty((beta.shape[0], len(meses)))
    for j, fm in enumerate(meses):
        t = strata.assign(date=fm)
        X, _, _ = diseno(t, geo, corte, tipo, levs)
        X = X.reindex(columns=cols, fill_value=0.0)
        if contrafactual: X[[c for c in cols if c in ('post', 'tiempo_post') or (c.startswith('p20') and len(c) == 5)]] = 0.0
        eta = X.values @ beta.T; out[:, j] = (strata.peso.values[:, None] / (1 + np.exp(-eta))).sum(0)
    return out

PERIODOS = {'2019': ('2019-01-01', '2019-12-01'), 'mar-dic 2020': ('2020-03-01', '2020-12-01'), '2021': ('2021-01-01', '2021-12-01'),
            '2022': ('2022-01-01', '2022-12-01'), '2023': ('2023-01-01', '2023-12-01')}
MESES = pd.date_range('2012-01-01', '2023-12-01', freq='MS')

def resumen_segmentado(t, y, geo, corte, tipo='segmentado', L=12, meses=MESES, strata=None, simular=True):
    g, X, C, levs, diag = ajustar(t, y, geo, corte, tipo, L)
    res = {'n_celdas': int(len(t)), 'n_defunciones': float(t.n.sum()), 'n_evento': float(t[y].sum()),
           'convergencia': bool(g.converged) if g is not None else False,
           'inferencial': diag['inferencial'], 'diagnosticos': diag}
    if not res['inferencial']:
        annual = t.assign(anio=pd.DatetimeIndex(t.date).year).groupby('anio')[[y,'n']].sum()
        res['descriptivos_anuales'] = {int(a): {'eventos':float(v[y]),'total':float(v['n']),
             'porcentaje':float(100*v[y]/v['n'])} for a,v in annual.iterrows()}
        return res, None
    cols = list(X.columns); b = g.params.values; ix = {c: i for i, c in enumerate(cols)}
    res['coeficientes'] = {col: float(value) for col,value in zip(cols,b)}
    def orc(terms):
        v = np.zeros(len(cols)); v[[ix[x] for x in terms]] = 1; e = float(v @ b); se = float(np.sqrt(v @ C @ v))
        return {'OR': np.exp(e), 'ic_inf': np.exp(e - 1.96 * se), 'ic_sup': np.exp(e + 1.96 * se), 'p': float(2 * stats.norm.sf(abs(e / se)))}
    if tipo == 'segmentado':
        res['OR'] = {'pendiente_pre_anual': orc(['tiempo']), 'cambio_nivel': orc(['post']), 'cambio_pendiente_anual': orc(['tiempo_post']),
                     'pendiente_post_anual': orc(['tiempo', 'tiempo_post'])}
        A = np.zeros((2, len(cols))); A[0, ix['post']] = 1; A[1, ix['tiempo_post']] = 1; bb = A @ b
        W = float(bb @ np.linalg.solve(A @ C @ A.T, bb)); res['wald_nivel_pendiente'] = {'chi2': W, 'p': float(stats.chi2.sf(W, 2))}
    elif tipo == 'anual_post':
        res['OR'] = {f'desvio_{y_}': orc([f'p{y_}']) for y_ in [2020, 2021, 2022, 2023]}
    if not simular: return res, None
    rng = np.random.default_rng(SEED); draws = np.vstack([b, rng.multivariate_normal(b, C, NSIM, method='cholesky')])
    ajus = estandarizar(draws, cols, strata, geo, meses, corte, tipo, False, levs)
    out = {'mensual': pd.DataFrame({'fecha': meses, 'ajustado': ajus[0], 'aj_inf': np.percentile(ajus[1:], 2.5, 0), 'aj_sup': np.percentile(ajus[1:], 97.5, 0)})}
    if tipo != 'anios':
        cf = estandarizar(draws, cols, strata, geo, meses, corte, tipo, True, levs)
        out['mensual']['contrafactual'] = cf[0]; out['mensual']['cf_inf'] = np.percentile(cf[1:], 2.5, 0); out['mensual']['cf_sup'] = np.percentile(cf[1:], 97.5, 0)
    per = {}
    for nom, (a, z) in PERIODOS.items():
        m = (meses >= a) & (meses <= z)
        if not m.any(): continue
        A_ = ajus[:, m].mean(1); e = {'ajustado_pct': 100 * A_[0], 'aj_ic': [100 * np.percentile(A_[1:], 2.5), 100 * np.percentile(A_[1:], 97.5)]}
        if tipo != 'anios':
            F = cf[:, m].mean(1); D = A_ - F
            e.update({'contrafactual_pct': 100 * F[0], 'cf_ic': [100 * np.percentile(F[1:], 2.5), 100 * np.percentile(F[1:], 97.5)],
                      'diferencia_pp': 100 * D[0], 'dif_ic': [100 * np.percentile(D[1:], 2.5), 100 * np.percentile(D[1:], 97.5)]})
        per[nom] = e
    m19 = (meses.year == 2019); m23 = (meses.year == 2023)
    D = ajus[:, m23].mean(1) - ajus[:, m19].mean(1)
    per['2023 menos 2019 (ajustado)'] = {'diferencia_pp': 100 * D[0], 'dif_ic': [100 * np.percentile(D[1:], 2.5), 100 * np.percentile(D[1:], 97.5)]}
    if tipo == 'anios':
        mm = [(meses.year == yy) for yy in sorted(set(meses.year))]
        per['anual'] = {int(yy): [100 * ajus[0, k].mean(), 100 * np.percentile(ajus[1:, k].mean(1), 2.5), 100 * np.percentile(ajus[1:, k].mean(1), 97.5)]
                        for yy, k in zip(sorted(set(meses.year)), mm)}
    res['periodos'] = per
    return res, out

t0 = time.time(); R = {'nsim': NSIM, 'semilla': SEED}; MENS = {}
base = w[w.residente & (w.anio_ocur <= 2023)]
# reconciliación descriptiva con entradas/lugar_defuncion_C50.csv
crudo = (w.groupby('anio_ocur').sitio.value_counts(normalize=True).unstack() * 100).round(1)
ref = pd.read_csv('../paquete_original/editorial/entradas/lugar_defuncion_C50.csv').set_index('anio_ocur')
R['reconciliacion_max_dif_pp'] = float((crudo[ref.columns] - ref).abs().max().max())
crudo_n = w.groupby(['anio_ocur', 'sitio']).size().unstack(); crudo_n.to_csv('../tablas/tablaS_lugar_conteos_anuales.csv')
strata = base.groupby(['ag', 'ent_resid']).size().rename('peso').reset_index(); strata['peso'] /= strata.peso.sum()
T = celdas(base); CUT = pd.Timestamp('2020-03-01')
for y in CATS:
    R[f'principal_{y}'], MENS[f'principal_{y}'] = resumen_segmentado(T, y, 'ent_resid', CUT, strata=strata)
    R[f'anual_post_{y}'], MENS[f'anual_post_{y}'] = resumen_segmentado(T, y, 'ent_resid', CUT, 'anual_post', strata=strata)
    R[f'anios_{y}'], MENS[f'anios_{y}'] = resumen_segmentado(T, y, 'ent_resid', CUT, 'anios', strata=strata)
    print(y, 'ok', round(time.time() - t0), flush=True)
# sensibilidades (desenlace primario: hogar; se repiten para pública y privada)
for y in ['Hogar', 'Unidad pública', 'Unidad privada']:
    R[f'sens_abril2020_{y}'], _ = resumen_segmentado(T, y, 'ent_resid', pd.Timestamp('2020-04-01'), strata=strata)
    R[f'sens_HAC6_{y}'], _ = resumen_segmentado(T, y, 'ent_resid', CUT, L=6, strata=strata)
    # Known places retain public road (10) and other place (12); only99 is unknown.
    kn = base[base.sitio_ocur != 99]; Tk = celdas(kn); sk = kn.groupby(['ag', 'ent_resid']).size().rename('peso').reset_index(); sk['peso'] /= sk.peso.sum()
    R[f'sens_lugar_conocido_{y}'], _ = resumen_segmentado(Tk, y, 'ent_resid', CUT, strata=sk)
    br = base.copy(); Tr = celdas(br, 'region'); sr = br.groupby(['ag', 'region']).size().rename('peso').reset_index(); sr['peso'] /= sr.peso.sum()
    R[f'sens_region_{y}'], _ = resumen_segmentado(Tr, y, 'region', CUT, strata=sr)
    nr = w[w.anio_ocur <= 2023].assign(ent2=lambda x: np.where(x.residente, x.ent_resid, 99)); Tn = celdas(nr, 'ent2')
    sn = nr.groupby(['ag', 'ent2']).size().rename('peso').reset_index(); sn['peso'] /= sn.peso.sum()
    R[f'sens_no_residentes_{y}'], _ = resumen_segmentado(Tn, y, 'ent2', CUT, strata=sn)
    b24 = w[w.residente]; M24 = pd.date_range('2012-01-01', '2024-12-01', freq='MS')
    for nom, pz in [('sin_ponderar', False), ('ponderado_completitud', True)]:
        R[f'sens_2024_{nom}_{y}'], _ = resumen_segmentado(celdas(b24, pesos=pz), y, 'ent_resid', CUT, meses=M24, strata=strata)
    print(y, 'sens ok', round(time.time() - t0), flush=True)
Ta = celdas(base, cat='sitio_alt')
for y in ['No especificado', 'Vía pública/otro']:
    R[f'sens_catalogo_alt_{y}'], _ = resumen_segmentado(Ta, y, 'ent_resid', CUT, strata=strata)
# multinomial (base hogar), EE agrupado por mes
Xi, mi, _ = diseno(base.assign(date=base.date.values).reset_index(drop=True), 'ent_resid', CUT)
yi = pd.Categorical(base.sitio.values, categories=CATS).codes
mn = sm.MNLogit(yi, Xi).fit(method='newton', maxiter=100, disp=0, cov_type='cluster', cov_kwds={'groups': mi})
Hmn = -mn.model.hessian(mn.params.values.ravel(order='F'))
Dmn = {'convergencia': bool(mn.mle_retvals.get('converged', False)), 'H_condicion': float(np.linalg.cond(Hmn)),
    'score_max_abs': float(np.max(np.abs(mn.model.score(mn.params.values.ravel(order='F'))))),
    'covarianza': 'Agrupada por mes; sensibilidad multinomial sin HAC temporal',
    'n_defunciones': int(len(base)), 'n_parametros': int(mn.params.size),
    'rango_diseno': int(np.linalg.matrix_rank(Xi.values)),
    'cov_eigen_min': float(np.linalg.eigvalsh(mn.cov_params().values).min())}
Dmn['inferencial'] = bool(Dmn['convergencia'] and Dmn['H_condicion'] < 1e10 and Dmn['score_max_abs']/len(base) < 1e-7 and Dmn['rango_diseno'] == Xi.shape[1] and Dmn['cov_eigen_min'] > -1e-10 and np.isfinite(mn.params.values).all() and np.isfinite(mn.cov_params().values).all())
R['diagnostico_multinomial'] = Dmn
R['sens_multinomial_base_hogar'] = {CATS[j + 1]: {p: {'RRR': float(np.exp(mn.params.loc[p, j])),
    'ic_inf': float(np.exp(mn.params.loc[p, j] - 1.96 * mn.bse.loc[p, j])), 'ic_sup': float(np.exp(mn.params.loc[p, j] + 1.96 * mn.bse.loc[p, j])),
    'p': float(mn.pvalues.loc[p, j])} for p in ['tiempo', 'post', 'tiempo_post']} for j in range(3)} if Dmn['inferencial'] else {}
# tasas estandarizadas por lugar (todas las residencias; mismos denominadores y pesos que la serie principal)
pm = poblacion_mensual(); tasas = {}
for nom, cats in [('Hogar', ['Hogar']), ('Unidad médica (pública + privada)', ['Unidad pública', 'Unidad privada']), ('Otro/NE', ['Otro/NE']), ('Total', CATS)]:
    cc = w[w.sitio.isin(cats)].groupby(['date', 'ag']).size().unstack(fill_value=0).reindex(IDX, fill_value=0)
    tasas[nom] = tee(cc, pm)
TS = pd.DataFrame(tasas); TS.index.name = 'fecha'; TS.to_csv('../resultados/tasas_estandarizadas_por_lugar_mensual.csv')
R['tasas_anuales_media_mensual'] = TS.groupby(TS.index.year).mean().round(3).to_dict()
R['tasas_metadata'] = {'2024': 'Observado, sin corrección por registro tardío; no se aplica factor global a sitios.',
    'periodo_ITS': '2012–2023', 'residencia': 'Todas, mismas WPP y pesos OMS del principal nacional',
    'sensibilidad_nacional_2024': 'La ITS nacional original conserva su corrección; análisis diferente.'}
R['diagnosticos_ITS_tasas'] = {}
for nom in tasas:
    y_ = TS[nom][TS.index < '2024-01-01']; X, tt = diseno_its(y_.index, CUT); m = sm.OLS(np.log(y_.values), X).fit()
    n, k = X.shape; C = dk_cov(X.values, X.values * m.resid.values[:, None], tt, np.linalg.inv(X.T.values @ X.values), 12, None) * n / (n - k)
    ix = {c: i for i, c in enumerate(X.columns)}; E = {}
    for p_, terms in [('pendiente_pre', ['tiempo']), ('salto_nivel', ['post']), ('cambio_pendiente', ['tiempo_post']), ('pendiente_post', ['tiempo', 'tiempo_post'])]:
        v = np.zeros(k); v[[ix[x] for x in terms]] = 1; E[p_] = contraste(m.params.values, C, v)
    R[f'ITS_tasa_{nom}'] = E
    R['diagnosticos_ITS_tasas'][nom] = {'n_meses': int(n), 'k': int(k), 'correccion': 'n/(n-k)',
        'ljung_box': acorr_ljungbox(m.resid.values,lags=[6,12,24],return_df=True).to_dict('index'),
        'autocorrelacion_1':float(np.corrcoef(m.resid.values[:-1],m.resid.values[1:])[0,1]),
        'residuo_sd_log':float(np.std(m.resid.values,ddof=k)),
        'nota':'Diagnóstico aproximado; dependencia residual no corregida en la media por usar HAC.'}
for k_, v in MENS.items():
    if v is not None: v['mensual'].to_csv(f'../resultados/lugar_mensual_{k_.replace("/", "_").replace(" ", "_")}.csv', index=False)
R['metadatos_v11'] = {'lugar_conocido': 'Solo excluye código99; conserva10 y12',
    'n_principal':int(len(base)), 'n_lugar_conocido':int((base.sitio_ocur!=99).sum()),
    'n_excluido_99':int((base.sitio_ocur==99).sum()),
    'estandarizacion_principal':'Distribución conjunta edad×entidad de residentes1–32,2012–2023, conservada de Claudev10',
    'inferencia_99':'Separación estructural; solo descriptivos y diagnósticos, sin OR/IC/p.'}
assert all(v.get('inferencial') for key,v in R.items() if key.startswith(('principal_','anual_post_','anios_')))
json.dump(R, open('../resultados/lugar_defuncion.json', 'w'), indent=1, ensure_ascii=False, default=float, allow_nan=False)
print('fin', round(time.time() - t0), 's; reconciliación', R['reconciliacion_max_dif_pp'])
