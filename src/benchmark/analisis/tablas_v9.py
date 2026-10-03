"""Tablas v8 a partir de resultados/ (P1-P5). Se ejecuta desde la raíz del proyecto. Salidas en documentos/tablas/."""
import numpy as np, pandas as pd, json
B = 'documentos/tablas/'; R = 'resultados/'
m = lambda x: f'{x:+.1f}'.replace('-', '−') if abs(x) >= 0.05 else '0.0'
s = lambda x: f'{x:.1f}'.replace('-', '−')
NAMES = {'asmr': 'Nacional, TEE', 'asmr_20-39': '20-39 años (TEE)', 'asmr_40-59': '40-59 años (TEE)', 'asmr_60+': '≥ 60 años (TEE)',
         'asmr_Norte': 'Norte (TEE)', 'asmr_Centro-Occidente': 'Centro-Occidente (TEE)', 'asmr_Centro': 'Centro (TEE)',
         'asmr_Sur-Sureste': 'Sur-Sureste (TEE)', 'deaths_adj': 'Nacional, defunciones', 'deaths_SS': 'Con seguridad social (defunciones)',
         'deaths_noSS': 'Sin seguridad social (defunciones)'}
O = pd.read_csv(R + 'p1_intervalos_mama_entre_series.csv')
# # ---------- Tabla 2 (v8): valores observados y esperados nacionales + diferencias por subgrupo ----------
EXA = pd.read_csv('entradas/tsfm_exceso_anual.csv'); ENSM = 'Ensamble (mediana)'
OLL = pd.read_csv(R + 'p1_intervalos_mama_loglineal.csv')
f0 = lambda v: f'{v:,.0f}'.replace(',', ' ')
YY = list(range(2020, 2024)); rows = []
def rr(lab, fun, f24=None):
    r = {'Serie': lab}
    for y in YY: r[str(y)] = fun(y)
    r['2024†'] = f24(2024) if f24 else fun(2024); rows.append(r)
ea = EXA[(EXA.model == ENSM) & (EXA.target == 'asmr')].set_index('year'); ed = EXA[(EXA.model == ENSM) & (EXA.target == 'deaths_adj')].set_index('year')
# La TEE anual del archivo es la SUMA de 12 tasas mensuales anualizadas; se presenta la media mensual (suma/12), por 100 000 mujeres-año
rows.append({'Serie': 'A. Nacional, mujeres ≥ 20 años', **{str(y): '' for y in YY}, '2024†': ''})
rr('TEE observada (por 100 000 mujeres-año)', lambda y: f"{ea.loc[y, 'observed'] / 12:.1f}")
rr('TEE esperada sin pandemia', lambda y: f"{ea.loc[y, 'expected'] / 12:.1f}")
dA = O[O.target == 'asmr'].set_index('year')
rr('Diferencia, % (IP 90 %)', lambda y: f"{m(dA.loc[y, 'pscore'])} ({s(dA.loc[y, 'lo_todas'])} a {s(dA.loc[y, 'hi_todas'])})", lambda y: m(dA.loc[y, 'pscore']))
rr('Defunciones observadas', lambda y: f0(ed.loc[y, 'observed']))
rr('Defunciones esperadas sin pandemia', lambda y: f0(ed.loc[y, 'expected']))
dD = O[O.target == 'deaths_adj'].set_index('year')
rr('Diferencia, % (IP 90 %)', lambda y: f"{m(dD.loc[y, 'pscore'])} ({s(dD.loc[y, 'lo_todas'])} a {s(dD.loc[y, 'hi_todas'])})", lambda y: m(dD.loc[y, 'pscore']))
dL = OLL[OLL.target == 'asmr'].set_index('year')
rr('TEE, modelo log-lineal (sensibilidad), % (IP 90 %)', lambda y: f"{m(dL.loc[y, 'pscore'])} ({s(dL.loc[y, 'lo'])} a {s(dL.loc[y, 'hi'])})", lambda y: m(dL.loc[y, 'pscore']))
rows.append({'Serie': 'B. Diferencia por subgrupo (TEE), % (IP 90 %)', **{str(y): '' for y in YY}, '2024†': ''})
for tg in ['asmr_20-39', 'asmr_40-59', 'asmr_60+', 'asmr_Norte', 'asmr_Centro-Occidente', 'asmr_Centro', 'asmr_Sur-Sureste']:
    d = O[O.target == tg].set_index('year')
    rr(NAMES[tg].replace(' (TEE)', ''), lambda y, d=d: f"{m(d.loc[y, 'pscore'])} ({s(d.loc[y, 'lo_todas'])} a {s(d.loc[y, 'hi_todas'])})", lambda y, d=d: m(d.loc[y, 'pscore']))
N4 = json.load(open('entradas/numeros_v4.json'))
jr = N4['joint_real']; rows.append({'Serie': 'Media de 22 series nacionales de cáncer (conteos), %', **{str(y): m(jr[str(y)]['mean_pct']) for y in range(2020, 2024)}, '2024†': '—'})
T2 = pd.DataFrame(rows); T2.to_csv(B + 'tabla2_v9.csv', index=False)
# significación con la calibración entre series
sig = O[(O.year <= 2023) & ((O.lo_todas > 0) | (O.hi_todas < 0))]
sig_nac = O[(O.year <= 2023) & ((O.lo_nacionales > 0) | (O.hi_nacionales < 0))]
sig_serie = O[(O.year <= 2023) & ((O.rango_serie_lo > 0) | (O.rango_serie_hi < 0))][['target', 'year', 'pscore', 'rango_serie_lo', 'rango_serie_hi']]
# ---------- Tabla S2B ----------
rows = []
for tg, nm in NAMES.items():
    d = O[O.target == tg].set_index('year')
    for y in range(2020, 2025):
        rows.append({'Serie': nm, 'Año': y, 'Diferencia (%)': m(d.loc[y, 'pscore']),
                     'Rango por serie (descriptivo)': f"{s(d.loc[y, 'rango_serie_lo'])} a {s(d.loc[y, 'rango_serie_hi'])}",
                     'IP 90 % entre series (todas)': (f"{s(d.loc[y, 'lo_todas'])} a {s(d.loc[y, 'hi_todas'])}" if y <= 2023 else f"({s(d.loc[y, 'lo_todas'])} a {s(d.loc[y, 'hi_todas'])})"),
                     'IP 90 % entre series (nacionales)': (f"{s(d.loc[y, 'lo_nacionales'])} a {s(d.loc[y, 'hi_nacionales'])}" if y <= 2023 else f"({s(d.loc[y, 'lo_nacionales'])} a {s(d.loc[y, 'hi_nacionales'])})"),
                     'Estrato de volumen': d.loc[y, 'vol']})
pd.DataFrame(rows).to_csv(B + 'tablaS2B_v9.csv', index=False)
D = pd.read_csv(R + 'p1_diagnostico_por_serie.csv'); D = D[D.target == 'asmr']
D2 = pd.DataFrame({'Año del horizonte': D.bloque, 'Errores de calibración (n)': D.n_errores, 'Rango requerido, entero superior de (n + 1) × 0.9': D.rango_requerido,
                   'Rango usado': D.rango_usado, 'Nivel nominal del máximo, n/(n+1) (%)': D.nivel_nominal_max.round(1),
                   'Correlación entre orígenes consecutivos': D.autocorr_lag1.map(lambda v: f'{v:.3f}' if v > 0.95 else f'{v:.2f}'), 'Sesgo medio (%)': D.sesgo_medio_pct.round(1),
                   'Máximo |error| (log)': D.max_abs.round(3)})
D2.to_csv(B + 'tablaS2C_diagnostico_por_serie.csv', index=False)
P = pd.read_csv(R + 'p1_placebo_calibracion_entre_series.csv'); P = P[P.vol.isin(['todos', '300-1000', '100-300', '25-100'])]  # v7: estratos 2012-2014
P.to_csv(B + 'tablaS2D_placebo_entre_series.csv', index=False)
# ---------- Tabla S9: falsas alarmas y detección (v6) ----------
R2 = pd.read_csv(R + 'p2_deteccion_resultados.csv'); D1 = pd.read_csv(R + 'p2_deteccion_diferencias_D1.csv').set_index('modelo')
MOD = ['LightGBM global', 'Ensamble (mediana)', 'TimesFM-2.5', 'Chronos-2', 'Chronos-Bolt', 'N-HiTS global', 'Log-lineal', 'ETS amortiguado', 'SARIMA', 'Ingenuo estacional']
nm = lambda x: x.replace(' global', '').replace(' (mediana)', '')
ci = lambda r, c: f"{s(r[c])} ({s(r[c + '_lo'])}-{s(r[c + '_hi'])})"
rows = []
P1 = R2[(R2.diseno.str.startswith('D1')) & (R2.bloque == '1-3')].set_index('modelo'); P3 = R2[(R2.diseno.str.startswith('D3')) & (R2.bloque == '1-3')].set_index('modelo')
pb = R2[R2.diseno.str.startswith('D1') & (R2.bloque != '1-3')].pivot(index='modelo', columns='anio_prueba', values='fa_sup')
for mo in MOD:
    r = P1.loc[mo]; r3 = P3.loc[mo]; dd = D1.loc[mo] if mo in D1.index else None
    rows.append({'Modelo': nm(mo), 'Falsas alarmas superiores, D1 (%)': ci(r, 'fa_sup'),
                 'Por año de prueba 2017 / 2018 / 2019 (%)': ' / '.join(s(pb.loc[mo, str(y)]) for y in (2017, 2018, 2019)),
                 'Umbral superior mediano (%)': f'{r.umbral_sup_mediano_pct:.0f}', 'Detección +10 %, D1 (%)': ci(r, 'det10'), 'Detección +20 %, D1 (%)': ci(r, 'det20'),
                 'Δ detección +20 % vs log-lineal (pp)': ('referencia' if dd is None else f"{m(dd.d_det20)} ({s(dd.d_det20_lo)} a {s(dd.d_det20_hi)})"),
                 'Bilateral 90 % (D3): superior / inferior (%)': f"{s(r3.fa_sup)} / {s(r3.fa_inf)}"})
pd.DataFrame(rows).to_csv(B + 'tablaS9_deteccion_v9.csv', index=False)
rows = []
R19 = pd.read_csv(R + 'p2_deteccion_estratos2012_2019_resultados.csv')
for lab, key, RR in [('D1 unilateral temporal (principal; estratos 2012-2014)', 'D1', R2), ('D1 con estratos 2012-2019 (sensibilidad)', 'D1', R19),
                     ('D2 unilateral sin ajuste', 'D2', R2), ('D3 bilateral 90 %', 'D3', R2), ('D0 factor k común (años compartidos; no independiente)', 'D0', R2)]:
    for mo in MOD:
        r = RR[(RR.diseno.str.startswith(key)) & (RR.bloque == '1-3') & (RR.modelo == mo)].iloc[0]
        rows.append({'Diseño': lab, 'Modelo': nm(mo), 'Falsas alarmas superiores (%)': s(r.fa_sup), 'Falsas alarmas inferiores (%)': s(r.fa_inf),
                     'Falsas alarmas bilaterales (%)': s(r.fa_bil), 'Detección +10 % (%)': s(r.det10), 'Detección +20 % (%)': s(r.det20)})
pd.DataFrame(rows).to_csv(B + 'tablaS9C_disenos_deteccion.csv', index=False)
dl = pd.read_csv(R + 'p2_deteccion_delta_D1.csv'); dl['modelo'] = dl.modelo.map(nm)
dlp = dl.pivot(index='modelo', columns='anio_ajuste', values='delta').reindex([nm(x) for x in MOD]).reset_index()
dlp.columns = ['Modelo'] + [f'δ (ajuste {c})' for c in dlp.columns[1:]]
for c in dlp.columns[1:]: dlp[c] = dlp[c].map(lambda v: f'{v:+.3f}'.replace('-', '−'))
dlp.to_csv(B + 'tablaS9D_delta.csv', index=False)
PC = pd.read_csv(R + 'p2_deteccion_control_positivo.csv'); PC['model'] = PC.model.map(nm)
PC = PC.set_index('model').reindex([nm(x) for x in MOD]).reset_index()
PC = PC.rename(columns={'model': 'Modelo', 'n': 'Series-año', 'n_series': 'Series', 'deteccion': 'Detección (%)'}); PC['Detección (%)'] = PC['Detección (%)'].map(s)
PC.to_csv(B + 'tablaS9B_control_positivo_v9.csv', index=False)
# ---------- Tabla S10: Friedman ----------
T = pd.read_csv(R + 'p4_friedman_sensibilidad.csv')
lab = {'base_por_serie': 'Principal: una observación por serie', 'conglomerado_grupo_N28_rango_de_rangos_medios': 'Conglomerado: grupo de causa',
       'conglomerado_causa_sexo': 'Conglomerado: causa × sexo', 'estatal_sin_todas_las_causas': 'Estatales, sin «todas las causas» (sin anidamiento)',
       'nivel_Nacional': 'Solo nacionales', 'nivel_Nacional×edad': 'Solo nacionales × edad', 'nivel_Estatal': 'Solo estatales', 'nivel_Estatal×edad': 'Solo estatales × edad'}
T['analisis'] = T.analisis.map(lab); T = T[['analisis', 'N', 'CD'] + MOD]
T.columns = ['Análisis', 'N', 'DC'] + [c.replace(' global', '').replace(' (mediana)', '') for c in MOD]
# v7: tabla transpuesta (modelos en filas) para que quepa sin cortar palabras
SH = {'Principal: una observación por serie': 'Principal (por serie)', 'Conglomerado: grupo de causa': 'Grupo de causa', 'Conglomerado: causa × sexo': 'Causa × sexo',
      'Estatales, sin «todas las causas» (sin anidamiento)': 'Sin anidamiento', 'Solo nacionales': 'Nacionales', 'Solo nacionales × edad': 'Nacionales × edad',
      'Solo estatales': 'Estatales', 'Solo estatales × edad': 'Estatales × edad'}
Tt = T.set_index('Análisis').T; Tt.columns = [SH.get(c, c) for c in Tt.columns]
Tt = Tt.astype(float).map(lambda v: f'{v:.2f}'); Tt.loc['N'] = [f'{int(float(v))}' for v in Tt.loc['N']]
Tt.index.name = 'Modelo'; Tt.reset_index().to_csv(B + 'tablaS10_friedman_sensibilidad.csv', index=False)
PB = pd.DataFrame(json.load(open(R + 'p4_friedman_sensibilidad.json'))['pares_bootstrap_grupo'])
PB.to_csv(B + 'tablaS10B_pares_bootstrap.csv', index=False)
# ---------- Tabla S11 (v8): déficit, escenarios de desplazamiento temporal y causas mal definidas ----------
C3 = json.load(open(R + 'p3_cosecha.json')); E = pd.read_csv(R + 'p3_escenarios_cosecha.csv')
F0 = lambda v: f'{v:,.0f}'.replace(',', ' ')
rows = []
for lab_, k in [('2021-2023 (principal)', '2021_2023'), ('2021-2024 (sensibilidad)', '2021_2024')]:
    ref = E[E.variante == 'referencia']
    rows.append({'Ventana': lab_, 'Déficit puntual en conteos': f"{C3[f'deficit_conteos_{k}']:,.1f}".replace(',', ' '),
                 'Déficit aproximado desde la TEE': f"{C3[f'deficit_TEE_{k}']:,.1f}".replace(',', ' '),
                 'Escenario de referencia, RR 1-3 y f 25-60 %': f"{F0(ref[f'A_{k}'].min())}-{F0(ref[f'A_{k}'].max())}",
                 'Todas las variantes': f"{F0(E[f'A_{k}'].min())}-{F0(E[f'A_{k}'].max())}",
                 'C50 adicional estimada en causas mal definidas vs 2017-2019': f"{C3[f'c50_oculto_exceso_vs_2017_19_{k}']:.0f}",
                 'Mal definidas / déficit en conteos (%)': f"{C3[f'c50_oculto_exceso_sobre_deficit_conteos_{k}_pct']:.1f}"})
pd.DataFrame(rows).to_csv(B + 'tablaS11_cosecha.csv', index=False)
# S11B: rejilla de escenarios (referencia) y variantes de un factor a la vez
ref = E[E.variante == 'referencia']; g = []
for RR in sorted(E.RR.unique()):
    r = {'RR supuesto': f'{RR:.1f}', 'Mujeres con cáncer de mama entre las defunciones por COVID-19 (H)': F0(ref[ref.RR == RR].H.iloc[0])}
    for f4 in sorted(E.f_2021_2024.unique()):
        a = ref[(ref.RR == RR) & (ref.f_2021_2024 == f4)].A_2021_2023.iloc[0]
        r[f'A 2021-2023, f = {f4*100:.0f} %'] = F0(a)
    vv = E[(E.RR == RR)]
    r['Intervalo con todas las variantes'] = f'{F0(vv.A_2021_2023.min())}-{F0(vv.A_2021_2023.max())}'
    g.append(r)
pd.DataFrame(g).to_csv(B + 'tablaS11B_escenarios_cosecha.csv', index=False)
# S11C: tabla de parámetros (fuente, población, medida, transformación, valores, justificación)
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'documentos'))
from referencias import REFS as _REFS
REF_ZEWDE = next(i for i, r in enumerate(_REFS, 1) if r.startswith('Zewde'))
P = [
 ['D: defunciones femeninas por COVID-19, ≥ 20 años, 2020-2022, por edad', 'INEGI, estadísticas de defunciones (código U07 como causa básica)', 'Mujeres fallecidas en México', 'Conteo', 'Ninguna', 'Dato observado', 'Registro nacional'],
 ['P5: prevalencia a 5 años de cáncer de mama', 'GLOBOCAN 2024, ficha de México (103 771; 154.0 por 100 000 mujeres)', 'Mujeres en México, 2024', 'Casos prevalentes a 5 años', 'kP = 1.00 o 0.90 para aproximar 2020-2021', 'Supuesto (kP)', 'La cifra ya corresponde a mujeres; no se aplica factor por sexo'],
 ['a: distribución por edad de los casos prevalentes', 'Aproximación con la edad de las defunciones C50 (INEGI 2017-2019)', 'Mujeres fallecidas por C50', 'Proporción por grupo de edad', 'a = w·(edad de las defunciones C50) + (1 − w)·(la misma, un grupo quinquenal más joven); w = 0.5 (referencia), 1 (edad de las defunciones C50) o 0 (≈ 5 años más joven)', 'Supuesto', 'Los casos prevalentes son más jóvenes que las defunciones; sin dato nacional'],
 ['RR: riesgo de morir por COVID-19, mujeres con frente a sin cáncer de mama, en la población', f'No se identificó una estimación directamente transferible. Zewde et al. (ref. {REF_ZEWDE}): OR 1.93 y HR 1.54 de mortalidad intrahospitalaria, pacientes hospitalizados con cualquier cáncer', 'No equivalente', 'OR/HR intrahospitalarios', 'No se transforma (OR→RR exige un riesgo basal no disponible)', 'Escenarios: 1.0, 1.5, 2.0 y 3.0', '1.0 = sin riesgo adicional'],
 ['U: subregistro de defunciones por COVID-19', 'Sin fuente específica', '—', '—', '—', 'U = 1.0 o 1.4', 'Referencia 1.0 (solo defunciones registradas)'],
 ['f: fracción que habría muerto por cáncer de mama en la ventana', 'Sin fuente específica', '—', '—', 'f(2021-2023) = 1 − (1 − f)^(3/4) (riesgo constante)', 'f(2021-2024) = 0.25 o 0.60', 'Depende del estadio, que no se conoce'],
 ['Déficit puntual (denominador)', 'Este estudio: Σ(esperadas − observadas), ensamble', 'Mujeres ≥ 20 años', 'Defunciones', 'Ninguna', 'Estimación puntual', 'Sus intervalos anuales incluyen el cero'],
]
pd.DataFrame(P, columns=['Parámetro', 'Fuente', 'Población', 'Medida original', 'Transformación', 'Valores usados', 'Justificación / nota']).to_csv(B + 'tablaS11C_parametros_cosecha.csv', index=False)
# ---------- Tabla S12: derechohabiencia ----------
dh = pd.read_csv(R + 'p5_derechohab_codigos_por_anio.csv', index_col=0); dg = pd.read_csv(R + 'p5_derechohab_grupos_por_anio_pct.csv', index_col=0)
dh.index.name = 'Código'; dg.index.name = 'Grupo'
dh.to_csv(B + 'tablaS12A_derechohab_codigos.csv'); dg.to_csv(B + 'tablaS12B_derechohab_grupos.csv')
cat = pd.read_csv(R + 'p5_catalogos_derechohab_por_anio.csv', index_col=0, dtype=str)
per = []; prev = None; ini = None
cols = list(cat.columns)
for i_, y in enumerate(cols):
    cur = tuple(cat[y].fillna('—'))
    if cur != prev:
        if prev is not None: per.append((ini, cols[i_ - 1], prev))
        ini, prev = y, cur
per.append((ini, cols[-1], prev))
C12 = pd.DataFrame({f'{a}-{b}' if a != b else a: list(v) for a, b, v in per}, index=cat.index); C12.index.name = 'Código'
C12.to_csv(B + 'tablaS12C_catalogos_inegi.csv')
# ---------- Tablas S2E-S2G: auditoría de los intervalos de mama ----------
cp = pd.read_csv(R + 'p1b_composicion_190.csv'); cp.columns = ['Fuente', 'Año del horizonte', 'Series', 'Orígenes', 'Errores', 'Desde', 'Hasta']
cp.to_csv(B + 'tablaS2E_composicion_190.csv', index=False)
pl = pd.read_csv(R + 'p1b_pool_calibracion.csv'); pl = pl[pl.vol.isin(['25-100', '100-300', '300-1000'])]
pl['% con C50 o todas las causas'] = (pl.n_c50_o_todas / pl.n * 100).round(1)
pl.columns = ['Año del horizonte', 'Estrato (defunciones/mes)', 'Errores de calibración', 'De series C50 o «todas las causas»', '% con C50 o todas las causas']
pl.to_csv(B + 'tablaS2F_pool_calibracion.csv', index=False)
sp = pd.read_csv(R + 'p1b_comprobacion_separada.csv')
sp = pd.DataFrame({'Año del horizonte': sp.bloque, 'Serie de mama': sp.target.map(lambda t: NAMES.get(t, t)),
                   'Variable': sp.variable.map({'tasa estandarizada': 'tasa', 'defunciones': 'conteo'}), 'Año de prueba': sp.anio_prueba,
                   'Años de calibración': sp.anios_calibracion.str.replace('[', '').str.replace(']', ''), 'Estrato': sp.vol, 'n': sp.n_cal,
                   'Umbral': sp.q.map(lambda v: f'{v:.3f}'), 'Error': sp.error.map(lambda v: f'{v:+.3f}'.replace('-', '−')),
                   'Fuera': sp.fuera.map({True: 'sí', False: 'no'})})
sp.to_csv(B + 'tablaS2G_comprobacion_separada.csv', index=False)
json.dump(dict(sig_todas=sig[['target', 'year']].values.tolist(), sig_nacionales=sig_nac[['target', 'year']].values.tolist(),
               sig_rango_serie=sig_serie.round(2).values.tolist()), open(R + 'p1_significacion.json', 'w'), ensure_ascii=False, indent=1)
print(T2.to_string()); print('sig entre series:', sig[['target', 'year']].values.tolist(), sig_nac[['target', 'year']].values.tolist())
print(sig_serie.round(1).to_string())

# ---------- Tabla S2H (v8): validación específica de la TEE nacional de mama (MASE) ----------
BM = pd.read_csv('entradas/tsfm_backtest_mensual.csv'); BM = BM[BM.target == 'asmr']
W = BM.pivot(index='origin', columns='model', values='MASE'); LLn = 'Log-lineal (tendencia+mes)'
ORD = ['Log-lineal (tendencia+mes)', 'Ensamble (mediana)', 'Chronos-Bolt', 'Chronos-2', 'Naive estacional', 'TimesFM-2.5', 'ETS amortiguado', 'SARIMA']
NM = {'Log-lineal (tendencia+mes)': 'Log-lineal', 'Ensamble (mediana)': 'Ensamble', 'Naive estacional': 'Ingenuo estacional'}
g = BM.groupby('model').agg(media=('MASE', 'mean'), mediana=('MASE', 'median'), mn=('MASE', 'min'), mx=('MASE', 'max'), cob=('cov80', 'mean')).loc[ORD]
S2H = pd.DataFrame({'Modelo': [NM.get(i, i) for i in g.index], 'MASE medio': g.media.map(lambda v: f'{v:.3f}'), 'MASE mediano': g.mediana.map(lambda v: f'{v:.3f}'),
                    'Mínimo-máximo': [f'{a:.2f}-{b:.2f}' for a, b in zip(g.mn, g.mx)], 'Cobertura observada del IP 80 % (%)': g.cob.map(lambda v: f'{v:.1f}'),
                    'Orígenes con MASE menor que el log-lineal (de 9)': ['—' if i == LLn else str(int((W[i] < W[LLn]).sum())) for i in g.index]})
S2H.to_csv(B + 'tablaS2H_mase_mama.csv', index=False)
