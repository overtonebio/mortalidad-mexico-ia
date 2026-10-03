"""P3 (v8). Déficit puntual de cáncer de mama y escenarios explícitos de desplazamiento temporal de la mortalidad («cosecha»).
Déficit principal: Σ(esperadas − observadas) de la serie de defunciones (deaths_adj, ensamble); sensibilidad: P-score de la TEE aplicado a
las defunciones observadas. La incertidumbre del déficit NO se propaga a ningún cociente.
Numerador (v8): ESCENARIOS, no estimación. Número esperado de mujeres con cáncer de mama entre las defunciones por COVID-19 de 2020-2022:
  H = Σ_edad D_e · U · π_e·RR / (1 − π_e + π_e·RR),  π_e = kP · P5 · a_e / N_e
  (D_e: defunciones femeninas por COVID-19 ≥ 20 años, INEGI; N_e: población femenina WPP 2024 media 2020-2021; P5 = 103 771, prevalencia a
  5 años de cáncer de mama en mujeres, GLOBOCAN 2024; a_e: distribución por edad de los casos prevalentes, aproximada; RR: cociente de riesgos
  de morir por COVID-19 en mujeres con y sin cáncer de mama en la población, SIN estimación transferible disponible; U: factor de subregistro).
  A = H · f, con f = fracción que habría fallecido por cáncer de mama en la ventana (2021-2024; para 2021-2023, 1 − (1 − f)^(3/4)).
Revisión v8: (1) el RR de v4-v7, lognormal centrado en 1.9 con límites 1.3-2.6, no procede de la referencia citada (Zewde et al.:
OR 1.93 y HR 1.54 de mortalidad intrahospitalaria en pacientes hospitalizados con cualquier cáncer), que no estima el cociente
poblacional que exige la fórmula; no se convierte OR en RR porque se necesitaría un riesgo basal no disponible. (2) Se retira el factor
U(0.98, 0.995) aplicado como «solo mujeres»: la cifra de GLOBOCAN ya corresponde a mujeres (154.0 por 100 000 mujeres).
(3) Los demás factores (periodo, edad, subregistro, f) no tienen fuente documentada y se tratan como escenarios.
La configuración previa (Monte Carlo v4-v7) se recalcula solo para trazabilidad (clave 'configuracion_previa_v7').
Causas mal definidas: entradas/subregistro_resultados.csv (defunciones C50 ESTIMADAS por el clasificador frente a 2017-2019).
Salidas: resultados/p3_cosecha.json, resultados/p3_escenarios_cosecha.csv"""
import numpy as np, pandas as pd, json
d = pd.read_csv('datos/c50_raw_2012_2024.csv.gz', dtype=str)
for c in ['sexo', 'edad', 'mes_ocurr', 'anio_ocur']: d[c] = pd.to_numeric(d[c], errors='coerce')
d['age'] = np.where((d.edad >= 4000) & (d.edad < 4998), d.edad - 4000, np.nan)
w = d[(d.sexo == 2) & (d.age >= 20) & d.mes_ocurr.between(1, 12) & d.anio_ocur.between(2012, 2024)].copy()
w['ag'] = (np.minimum(w.age, 85) // 5 * 5).astype(int)
idx = pd.date_range('2012-01-01', '2024-12-01', freq='MS')
p = pd.read_csv('datos/wpp_mex_female_age5.csv'); p['ag'] = np.minimum(p.AgeGrpStart, 85)
p = p[p.ag >= 20].groupby(['Time', 'ag']).PopFemale.sum().unstack() * 1000; p.index = pd.to_datetime(p.index.astype(str) + '-07-01')
pm = p.reindex(p.index.union(idx + pd.Timedelta(days=14))).interpolate('time').reindex(idx + pd.Timedelta(days=14)); pm.index = idx
ags = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
covt = pd.read_csv('datos/covid_mujeres_edad.csv', index_col=0); covt.columns = covt.columns.astype(int)
c50age = w[w.anio_ocur.between(2017, 2019)].groupby('ag').size(); c50age = c50age / c50age.sum()
D = covt.loc[2020:2022, ags].sum().values.astype(float); popa = pm.loc['2020-07-01':'2021-12-01', ags].mean().values
age_old = c50age.reindex(ags).fillna(0).values; age_young = np.r_[age_old[1:], 0]; age_young = age_young / age_young.sum()
P5 = 103771.0
# w = peso de la distribución por edad de las defunciones C50 (age_old); 1 − w = peso de la misma distribución desplazada un grupo
# quinquenal más joven (age_young). w = 1: edad de las defunciones C50; w = 0: ≈ 5 años más joven. (v9: etiquetas de las variantes corregidas.)
def H_of(RR, kP=1.0, w=0.5, U=1.0):
    a = w * age_old + (1 - w) * age_young; pi = kP * P5 * a / popa
    return float((D * U * pi * RR / (1 - pi + pi * RR)).sum())
f3 = lambda f4: 1 - (1 - f4) ** 0.75
# configuración previa (v4-v7), solo trazabilidad
rng = np.random.default_rng(7); Nmc = 20000
P5mc = P5 * rng.uniform(0.98, 0.995, Nmc) * rng.uniform(0.88, 0.95, Nmc); wmix = rng.uniform(0, 1, Nmc)[:, None]
agedist = wmix * age_old + (1 - wmix) * age_young
pi = (P5mc[:, None] * agedist) / popa; RRmc = np.exp(rng.normal(np.log(1.9), np.log(2.6 / 1.3) / (2 * 1.96), Nmc))[:, None]
Umc = rng.uniform(1.0, 1.4, Nmc)[:, None]
Hh = (D * Umc * pi * RRmc / (1 - pi + pi * RRmc)).sum(1); ff = rng.uniform(0.25, 0.60, Nmc); A4 = Hh * ff
A_pub = np.load('entradas/cosecha_simulaciones.npy'); assert np.allclose(A4, A_pub), 'no reproduce cosecha_simulaciones.npy'
A3 = Hh * (1 - (1 - ff) ** 0.75)
# déficit puntual
S = pd.read_csv('datos/series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
ann = S.groupby(S.index.year)[['deaths', 'deaths_adj']].sum()
EX = pd.read_csv('entradas/tsfm_exceso_anual.csv'); ee = EX[(EX.target == 'asmr') & (EX.model == 'Ensamble (mediana)')].set_index('year')
def defi(yrs, col):
    o = ann.loc[yrs, col].values; pp = ee.loc[yrs, 'pscore'].values; return float((o / (1 + pp / 100) - o).sum())
ec = EX[(EX.target == 'deaths_adj') & (EX.model == 'Ensamble (mediana)')].set_index('year')
def defc(yrs): return float((ec.loc[yrs, 'expected'] - ec.loc[yrs, 'observed']).sum())
res = {}
for lab, yrs in [('2021_2023', [2021, 2022, 2023]), ('2021_2024', [2021, 2022, 2023, 2024])]:
    for col in ['deaths', 'deaths_adj']:
        res[f'deficit_puntual_{lab}_{col}'] = defi(yrs, col)
    dc = defc(yrs); dt = res[f'deficit_puntual_{lab}_deaths_adj']; res[f'deficit_conteos_{lab}'] = dc; res[f'deficit_TEE_{lab}'] = dt
# ---------- escenarios explícitos ----------
RRs = [1.0, 1.5, 2.0, 3.0]; F4 = [0.25, 0.60]
rows = []
for RR in RRs:
    for f4 in F4:
        for (kP, w, U, lab) in [(1.0, 0.5, 1.0, 'referencia'), (0.90, 0.5, 1.0, 'prevalencia ×0.90'), (1.0, 0.0, 1.0, 'edad: 5 años más joven'),
                                (1.0, 1.0, 1.0, 'edad: defunciones C50'), (1.0, 0.5, 1.4, 'subregistro COVID ×1.4')]:
            h = H_of(RR, kP, w, U)
            rows.append(dict(RR=RR, f_2021_2024=f4, f_2021_2023=f3(f4), variante=lab, kP=kP, w_edad=w, U=U, H=h, A_2021_2023=h * f3(f4), A_2021_2024=h * f4,
                             pct_deficit_conteos_2021_2023=h * f3(f4) / res['deficit_conteos_2021_2023'] * 100,
                             pct_deficit_conteos_2021_2024=h * f4 / res['deficit_conteos_2021_2024'] * 100))
E = pd.DataFrame(rows); E.to_csv('resultados/p3_escenarios_cosecha.csv', index=False)
ref = E[E.variante == 'referencia']
res['escenarios'] = dict(RR=RRs, f_2021_2024=F4, P5=P5,
    A_2021_2023_ref_min=float(ref.A_2021_2023.min()), A_2021_2023_ref_max=float(ref.A_2021_2023.max()),
    A_2021_2023_min=float(E.A_2021_2023.min()), A_2021_2023_max=float(E.A_2021_2023.max()),
    A_2021_2024_min=float(E.A_2021_2024.min()), A_2021_2024_max=float(E.A_2021_2024.max()),
    H_ref={str(r): H_of(r) for r in RRs},
    pct_2021_2023_min=float(E.pct_deficit_conteos_2021_2023.min()), pct_2021_2023_max=float(E.pct_deficit_conteos_2021_2023.max()))
res['configuracion_previa_v7'] = dict(A_2021_2023_p2_5_50_97_5=np.percentile(A3, [2.5, 50, 97.5]).round(2).tolist(),
    A_2021_2024_p2_5_50_97_5=np.percentile(A4, [2.5, 50, 97.5]).round(2).tolist(),
    cociente_mediana_2021_2023_pct=float(np.median(A3) / res['deficit_conteos_2021_2023'] * 100),
    nota='RR lognormal(1.9; 1.3-2.6) sin fuente que lo respalde; factor «solo mujeres» erróneo; retirada en v8')
sub = pd.read_csv('entradas/subregistro_resultados.csv').set_index('y')
for lab, yrs in [('2021_2023', [2021, 2022, 2023]), ('2021_2024', [2021, 2022, 2023, 2024])]:
    x = float(sub.loc[yrs, 'exceso_vs_2017_19'].sum()); res[f'c50_oculto_exceso_vs_2017_19_{lab}'] = x
    res[f'c50_oculto_exceso_sobre_deficit_conteos_{lab}_pct'] = x / res[f'deficit_conteos_{lab}'] * 100
    res[f'c50_oculto_exceso_sobre_deficit_TEE_{lab}_pct'] = x / res[f'deficit_TEE_{lab}'] * 100
res['anual'] = {int(y): dict(obs=float(ann.loc[y, 'deaths']), obs_adj=float(ann.loc[y, 'deaths_adj']), pscore_TEE=float(ee.loc[y, 'pscore']),
                             deficit_TEE=float(ann.loc[y, 'deaths_adj'] / (1 + ee.loc[y, 'pscore'] / 100) - ann.loc[y, 'deaths_adj']),
                             esperadas_conteos=float(ec.loc[y, 'expected']), deficit_conteos=float(ec.loc[y, 'expected'] - ec.loc[y, 'observed']),
                             c50_oculto_exceso=float(sub.loc[y, 'exceso_vs_2017_19'])) for y in range(2020, 2025)}
json.dump(res, open('resultados/p3_cosecha.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'anual'}, ensure_ascii=False, indent=1))
