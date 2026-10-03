"""Reúne en resultados/numeros_v9.json todas las cifras que cita el manuscrito v8 (fuente única).
Cifras publicadas (v3/v4, reproducidas desde fc_*.npz) + reanálisis v5 (v5/resultados)."""
import json, numpy as np, pandas as pd, os
# Fuente única del benchmark: resultados/benchmark/numeros_repro.json (reproducido desde pronosticos/publicados).
# Las claves que el reproductor no calcula (h48, mama48, sig2023, friedman_serie, fallback_sarima_ets) se toman de
# entradas/numeros_v4.json; se verifica que las claves comunes coinciden exactamente.
NR = json.load(open('resultados/benchmark/numeros_repro.json')); NV4 = json.load(open('entradas/numeros_v4.json'))
def _eq(a, b):
    if isinstance(a, dict): return all(k in b and _eq(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return abs(a - b) < 1e-9
    return a == b
assert all(_eq(NR[k], NV4[k]) for k in NR if k in NV4), 'numeros_repro difiere de numeros_v4'
N4 = dict(NR); N4.update({k: NV4[k] for k in NV4 if k not in NR}); N4['_procedencia'] = {'benchmark': 'resultados/benchmark/numeros_repro.json', 'solo_v4': sorted(k for k in NV4 if k not in NR)}
G4 = json.load(open('entradas/numeros_G.json'))
R = 'resultados/'; out = {'v4': N4, 'G': G4}
out['p1_diag'] = pd.read_csv(R + 'p1_diagnostico_por_serie.csv').query("target=='asmr'").set_index('bloque').to_dict('index')
O = pd.read_csv(R + 'p1_intervalos_mama_entre_series.csv')
out['mama'] = {f'{t}|{y}': r for (t, y), r in O.set_index(['target', 'year']).to_dict('index').items()}
P = pd.read_csv(R + 'p1_placebo_calibracion_entre_series.csv')
out['p1_placebo'] = {f"{r.variante}|{r.bloque}|{r.vol}": dict(FPR=r.FPR, n=r.n_prueba, sesgo=r.sesgo_prueba_pct) for r in P.itertuples()}
out['p1_sig'] = json.load(open(R + 'p1_significacion.json'))
R2 = pd.read_csv(R + 'p2_deteccion_resultados.csv')
out['p2'] = {k: R2[R2.diseno.str.startswith(k) & (R2.bloque == '1-3')].set_index('modelo').to_dict('index') for k in ('D0', 'D1', 'D2', 'D3')}
out['p2_bloque'] = {k: {f'{r.modelo}|{r.anio_prueba}': dict(fa_sup=r.fa_sup, det20=r.det20) for r in R2[R2.diseno.str.startswith(k) & (R2.bloque != '1-3')].itertuples()} for k in ('D1',)}
R19 = pd.read_csv(R + 'p2_deteccion_estratos2012_2019_resultados.csv')
out['p2_estratos2012_2019'] = {r.modelo: r._asdict() for r in R19[R19.diseno.str.startswith('D1') & (R19.bloque.astype(str) == '1-3')].itertuples(index=False)}
D3_19 = R19[R19.diseno.str.startswith('D3') & (R19.bloque.astype(str) == '1-3')].set_index('modelo').fa_bil
D3_14 = R2[R2.diseno.str.startswith('D3') & (R2.bloque.astype(str) == '1-3')].set_index('modelo').fa_bil
out['d3_bil_dif_max_pp'] = float((D3_14 - D3_19).abs().max())
PL = pd.read_csv(R + 'p1_placebo_calibracion_entre_series.csv'); PLm = PL[(PL.variante.str.startswith('Mondrian')) & (PL.vol != 'todos')]
out['p1_placebo_rango'] = dict(estratos='2012-2014', fpr_min=float(PLm.FPR.min()), fpr_max=float(PLm.FPR.max()),
                               fpr_total={int(r.bloque): float(r.FPR) for r in PL[(PL.variante.str.startswith('Mondrian')) & (PL.vol == 'todos')].itertuples()})
out['p2_dif'] = pd.read_csv(R + 'p2_deteccion_diferencias_D1.csv').set_index('modelo').to_dict('index')
out['p2_pc'] = pd.read_csv(R + 'p2_deteccion_control_positivo.csv').set_index('model').to_dict('index')
out['p2_resumen'] = json.load(open(R + 'p2_deteccion_resumen.json'))
out['p3'] = json.load(open(R + 'p3_cosecha.json'))
out['p1b'] = json.load(open(R + 'p1b_resumen.json'))
pl = pd.read_csv(R + 'p1b_pool_calibracion.csv'); out['p1b_pool'] = {f'{r.b}|{r.vol}': dict(n=int(r.n), n_c50=int(r.n_c50_o_todas)) for r in pl.itertuples()}
se = pd.read_csv(R + 'p1b_sensibilidad_sin_C50.csv'); out['p1b_sens'] = {f'{r.target}|{r.year}': dict(lo=r.lo_sin_C50_ni_todas, hi=r.hi_sin_C50_ni_todas) for r in se.itertuples()}
out['dh_sin_catalogo'] = pd.read_csv(R + 'p5_codigos_sin_catalogo.csv').to_dict('records')
out['p4'] = json.load(open(R + 'p4_friedman_sensibilidad.json'))
out['lora'] = pd.read_csv(R + 'p4b_lora_armonizado.csv').set_index('variante').to_dict('index')
dg = pd.read_csv(R + 'p5_derechohab_grupos_por_anio_pct.csv', index_col=0); out['dh_grupos'] = dg.to_dict()
dc = pd.read_csv(R + 'p5_derechohab_codigos_por_anio.csv', index_col=0); out['dh_codigos'] = dc.to_dict()
S = pd.read_csv('datos/series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
a = S.groupby(S.index.year)[['deaths', 'deaths_adj']].sum(); out['mama_anual'] = a.to_dict('index')
# errores de validación de mama (11 series) que superan el cuantil entre series
out['breast_backtest_exceed'] = json.load(open(R + 'p1_resumen.json'))['backtest_mama_supera_cuantil_entre_series']
import statsmodels.api as sm
y = S.loc['2012':'2019', 'asmr']; tt = np.arange(len(y)) / 12
Xm = pd.get_dummies(y.index.month, drop_first=True).astype(float).values
fit = sm.OLS(np.log(y.values), sm.add_constant(np.c_[tt, Xm])).fit()
ci = fit.conf_int()[1]
out['tendencia_TEE_2012_2019'] = dict(apc=float((np.exp(fit.params[1]) - 1) * 100), lo=float((np.exp(ci[0]) - 1) * 100), hi=float((np.exp(ci[1]) - 1) * 100),
                                      extremos_pct=float(((S.loc['2019', 'asmr'].mean() / S.loc['2012', 'asmr'].mean()) ** (1 / 7) - 1) * 100))
# ---------- v8 ----------
out['mase_mama'] = json.load(open(R + 'p1_resumen.json'))['mase_mama_tee_nacional']
BM = pd.read_csv('entradas/tsfm_backtest_mensual.csv'); BM = BM[BM.target == 'asmr']
out['mase_mama']['mediana'] = BM.groupby('model').MASE.median().round(4).to_dict()
out['mase_mama']['cobertura80'] = BM.groupby('model').cov80.mean().round(1).to_dict() if 'cov80' in BM.columns else None
LLi = pd.read_csv(R + 'p1_intervalos_mama_loglineal.csv'); out['mama_loglineal'] = {f'{r.target}|{r.year}': dict(pscore=r.pscore, lo=r.lo, hi=r.hi) for r in LLi.itertuples()}
EXA = pd.read_csv('entradas/tsfm_exceso_anual.csv'); EXA = EXA[EXA.model == 'Ensamble (mediana)']
out['mama_obs_esp'] = {f'{r.target}|{r.year}': dict(obs=r.observed / (12 if r.target == 'asmr' else 1), esp=r.expected / (12 if r.target == 'asmr' else 1))
                       for r in EXA[EXA.target.isin(['asmr', 'deaths_adj'])].itertuples()}
# posición del cáncer de mama entre los 13 grupos de cáncer en mujeres (series nacionales del conjunto, todas las edades)
Cp = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz'); Cp = Cp[(Cp.level == 'Nacional') & (Cp.sex == 2)]
Gc = pd.read_csv('entradas/grupos_CIE10_28.csv'); canc = Gc[Gc.Tipo == 'Cánceres'].Grupo.tolist()
yr = {y: [str(i) for i in range((y - 2012) * 12, (y - 2012) * 12 + 12)] for y in range(2012, 2025)}
tc = pd.DataFrame({y: Cp[v].sum(1).values for y, v in yr.items()}, index=Cp.grp.values).loc[lambda d: d.index.isin(canc)]
rk = tc.rank(ascending=False).loc['C. mama']
out['mama_primera_causa_cancer_mujeres'] = dict(anios_rango_1=int((rk == 1).sum()), n_anios=len(rk), n_grupos=int(len(tc)),
    segunda_2023=tc[2023].sort_values(ascending=False).index[1], mama_2023=int(tc.loc['C. mama', 2023]), segunda_2023_n=int(tc[2023].sort_values(ascending=False).iloc[1]))
json.dump(out, open('resultados/numeros_v9.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('ok', len(json.dumps(out, default=float)))
