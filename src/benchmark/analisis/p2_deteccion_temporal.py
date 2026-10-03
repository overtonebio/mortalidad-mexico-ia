"""P2 (v7: estratos de volumen fijados con 2012-2014; variable de entorno ESTRATOS=2012-2019 reproduce v6). Falsas alarmas y detección de exceso con separación temporal por horizonte.
Entrada: resultados/benchmark/metricas_por_serie.csv.gz (errores anuales firmados e = log((obs+1)/(pred+1)) por serie, origen,
modelo y año del horizonte b = 1..3; recalculados desde pronosticos/publicados con reproducir_benchmark.py) y
resultados/benchmark/atlas_repro.csv.gz (contrafactual 2020-2021 para los controles positivos).
Año calendario del error: año(origen) + b. Orígenes 2014, 2015 y 2016 → años 2015-2017, 2016-2018 y 2017-2019.
El entrenamiento de los modelos (datos previos a cada origen) es distinto de la calibración del detector (errores de orígenes anteriores).

Diseños:
 D0  «v5» (reconstruido para diagnóstico): cuantil bilateral |e| con origen 2014; UN factor k por modelo ajustado con el
     origen 2015 agrupando b = 1-3 (años 2016-2018); prueba con el origen 2016 (años 2017-2019). El factor k usa desenlaces de
     2017 y 2018, que también son años de prueba (b = 1 y 2) ⇒ no es independiente.
 D1  Principal v6, unilateral y temporal por horizonte: para cada b, umbral superior = cuantil conformal 95 % de e (firmado) con el
     origen 2014 (año 2014+b), por modelo y estrato de volumen; desplazamiento aditivo δ(modelo, b) = el menor que deja ≤ 5 % de
     falsas alarmas superiores con el origen 2015 (año 2015+b); prueba con el origen 2016 (año 2016+b). Cada umbral usa solo
     desenlaces anteriores a su año de prueba. Objetivo común de calibración: 5 % unilateral (no implica tasas observadas iguales).
 D2  Sensibilidad unilateral sin ajuste: cuantil conformal 95 % de e con los orígenes 2014 y 2015 (años 2014+b y 2015+b); prueba 2016+b.
 D3  Bilateral publicado (|e|, 90 %, orígenes 2014+2015, prueba 2016), informado con sus colas superior e inferior por separado.
Detección: exceso inyectado obs×(1+d), d = 0.10 y 0.20 ⇒ e + log(1+d) (aprox.; < 0.005 en log con ≥ 60 defunciones/año) > umbral superior.
IC 95 %: bootstrap por grupo de causa (28 grupos), 2 000 réplicas, umbrales fijos.
Controles positivos (diabetes, cardiopatía isquémica y todas las causas; series de todas las edades, nacionales y estatales; 2020 y
2021): umbral unilateral 95 % con los tres orígenes (b = 1 para 2020, b = 2 para 2021), solo con desenlaces previos a 2020.
Salidas: resultados/p2_deteccion_*.csv y p2_deteccion_resumen.json."""
import numpy as np, pandas as pd, json
M = pd.read_csv('resultados/benchmark/metricas_por_serie.csv.gz')
MODS = ['Ensamble (mediana)', 'LightGBM global', 'Chronos-2', 'TimesFM-2.5', 'Chronos-Bolt', 'N-HiTS global',
        'ETS amortiguado', 'Log-lineal', 'SARIMA', 'Ingenuo estacional']
A_UP = 0.05
def cq_up(e, a=A_UP):
    e = np.sort(np.asarray(e)); n = len(e); r = int(np.ceil((n + 1) * (1 - a))); return np.inf if r > n else e[r - 1]
def cq_abs(e, a=0.10):
    e = np.sort(np.abs(np.asarray(e))); n = len(e); r = int(np.ceil((n + 1) * (1 - a))); return np.inf if r > n else e[r - 1]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from estratos import vol_corpus
BASE = os.environ.get('ESTRATOS', '2012-2014'); SUF = '' if BASE == '2012-2014' else '_estratos' + BASE.replace('-', '_')
VOL = vol_corpus(pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0), BASE)
M['vol'] = M.sid.map(VOL)
L = pd.concat([M.assign(b=b, e=M[f'logerr_y{b}'])[['sid', 'origin', 'model', 'grp', 'level', 'vol', 'b', 'e']] for b in (1, 2, 3)])
L['anio'] = L.origin.str[:4].astype(int) + L.b
L = L[L.model.isin(MODS)]
yrs = L.groupby(['origin', 'b']).anio.first().unstack(); yrs.to_csv(f'resultados/p2_deteccion{SUF}_anios_por_origen.csv')
G = L.grp.unique()
rng = np.random.default_rng(2026)
def boot(df, cols, B=2000):
    gi = {g: np.flatnonzero(df.grp.values == g) for g in G}; V = df[cols].values.astype(float)
    bs = np.array([V[np.concatenate([gi[g] for g in rng.choice(G, len(G))])].mean(0) for _ in range(B)]) * 100
    return V.mean(0) * 100, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
def score(T, thr_up, thr_lo=None):
    T = T.copy(); T['fa_sup'] = T.e > thr_up
    T['fa_inf'] = (T.e < thr_lo) if thr_lo is not None else False
    T['fa_bil'] = T.fa_sup | T.fa_inf
    for d in (10, 20): T[f'det{d}'] = (T.e + np.log(1 + d / 100)) > thr_up
    return T
tr, tu, te = (L[L.origin == o] for o in ('2014-12', '2015-12', '2016-12'))
out = {}
# ---------------- D1 (principal) ----------------
q14 = tr.groupby(['model', 'vol', 'b']).e.apply(cq_up).rename('q').reset_index()
tu1 = tu.merge(q14, on=['model', 'vol', 'b']); te1 = te.merge(q14, on=['model', 'vol', 'b'])
GRID = np.round(np.arange(-0.30, 0.3001, 0.001), 3); delta = {}
for (m, b), g in tu1.groupby(['model', 'b']):
    fa = np.array([(g.e > g.q + d).mean() for d in GRID]); delta[(m, b)] = float(GRID[np.argmax(fa <= A_UP)])
te1['delta'] = [delta[(m, b)] for m, b in zip(te1.model, te1.b)]; te1['thr'] = te1.q + te1.delta
S1 = score(te1, te1.thr)
# ---------------- D2 (sin ajuste) ----------------
q1415 = pd.concat([tr, tu]).groupby(['model', 'vol', 'b']).e.apply(cq_up).rename('q').reset_index()
te2 = te.merge(q1415, on=['model', 'vol', 'b']); S2 = score(te2, te2.q)
# ---------------- D3 (bilateral publicado) ----------------
qa = pd.concat([tr, tu]).groupby(['model', 'vol', 'b']).e.apply(cq_abs).rename('q').reset_index()
te3 = te.merge(qa, on=['model', 'vol', 'b']); S3 = score(te3, te3.q, -te3.q)
# ---------------- D0 (v5 reconstruido) ----------------
qa14 = tr.groupby(['model', 'vol', 'b']).e.apply(cq_abs).rename('q').reset_index()
tu0 = tu.merge(qa14, on=['model', 'vol', 'b']); te0 = te.merge(qa14, on=['model', 'vol', 'b'])
K = np.round(np.arange(0.30, 3.001, 0.005), 3); kk = {}
for m, g in tu0.groupby('model'):
    fpr = np.array([(g.e.abs() > k * g.q).mean() for k in K]); kk[m] = float(K[np.argmax(fpr <= 0.10)])
te0['thr'] = te0.q * te0.model.map(kk); S0 = score(te0, te0.thr, -te0.thr)
cols = ['fa_sup', 'fa_inf', 'fa_bil', 'det10', 'det20']
rows = []
for lab, S in [('D1 unilateral temporal (principal)', S1), ('D2 unilateral sin ajuste', S2), ('D3 bilateral publicado', S3), ('D0 v5 reconstruido', S0)]:
    for m in MODS:
        for bsel in ['1-3', 1, 2, 3]:
            T = S[S.model == m] if bsel == '1-3' else S[(S.model == m) & (S.b == bsel)]
            pt, lo, hi = boot(T, cols, B=2000 if bsel == '1-3' else 500)
            r = dict(diseno=lab, modelo=m, bloque=str(bsel), anio_prueba=('2017-2019' if bsel == '1-3' else str(2016 + bsel)), n=len(T))
            for j, c in enumerate(cols): r[c] = pt[j]; r[c + '_lo'] = lo[j]; r[c + '_hi'] = hi[j]
            if 'thr' in T: r['umbral_sup_mediano_pct'] = float(np.median((np.exp(T.thr) - 1) * 100))
            else: r['umbral_sup_mediano_pct'] = float(np.median((np.exp(T.q) - 1) * 100))
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'resultados/p2_deteccion{SUF}_resultados.csv', index=False)
# diferencias pareadas frente al log-lineal (D1, b = 1-3), bootstrap por grupo
ref = S1[S1.model == 'Log-lineal'].set_index(['sid', 'b']); dif = []
for m in MODS:
    if m == 'Log-lineal': continue
    T = S1[S1.model == m].set_index(['sid', 'b']).loc[ref.index]
    D = np.c_[T.fa_sup.values.astype(float) - ref.fa_sup.values, T.det10.values.astype(float) - ref.det10.values, T.det20.values.astype(float) - ref.det20.values]
    grp = ref.grp.values; gi = {g: np.flatnonzero(grp == g) for g in G}
    bs = np.array([D[np.concatenate([gi[g] for g in rng.choice(G, len(G))])].mean(0) for _ in range(2000)]) * 100
    pt = D.mean(0) * 100
    dif.append(dict(modelo=m, d_fa_sup=pt[0], d_fa_sup_lo=np.percentile(bs[:, 0], 2.5), d_fa_sup_hi=np.percentile(bs[:, 0], 97.5),
                    d_det10=pt[1], d_det10_lo=np.percentile(bs[:, 1], 2.5), d_det10_hi=np.percentile(bs[:, 1], 97.5),
                    d_det20=pt[2], d_det20_lo=np.percentile(bs[:, 2], 2.5), d_det20_hi=np.percentile(bs[:, 2], 97.5)))
pd.DataFrame(dif).to_csv(f'resultados/p2_deteccion{SUF}_diferencias_D1.csv', index=False)
pd.DataFrame([dict(modelo=m, bloque=b, anio_ajuste=2015 + b, delta=d) for (m, b), d in delta.items()]).to_csv(f'resultados/p2_deteccion{SUF}_delta_D1.csv', index=False)
pd.DataFrame([dict(modelo=m, k=v) for m, v in kk.items()]).to_csv(f'resultados/p2_deteccion{SUF}_k_D0.csv', index=False)
# ---------------- controles positivos ----------------
A = pd.read_csv('resultados/benchmark/atlas_repro.csv.gz')
PCG = ['Diabetes mellitus', 'Cardiopatía isquémica', 'Todas las causas']
P = A[A.grp.isin(PCG) & A.year.isin([2020, 2021]) & (A.ab == 'Todas') & A.model.isin(MODS)].copy(); P['b'] = P.year - 2019; P['vol'] = P.sid.map(VOL)
q3 = L.groupby(['model', 'vol', 'b']).e.apply(cq_up).rename('q').reset_index()
P = P.merge(q3, on=['model', 'vol', 'b']); P['det'] = np.log((P.obs + 1) / (P.exp + 1)) > P.q
PC = P.groupby(['model']).agg(n=('sid', 'size'), n_series=('sid', 'nunique'), deteccion=('det', 'mean')).reset_index(); PC['deteccion'] *= 100
PC.to_csv(f'resultados/p2_deteccion{SUF}_control_positivo.csv', index=False)
json.dump(dict(grupos_control_positivo=PCG, niveles_control_positivo=sorted(P.level.unique().tolist()),
               n_series_control_positivo=int(P.sid.nunique()), anios=yrs.to_dict()), open(f'resultados/p2_deteccion{SUF}_resumen.json', 'w'), ensure_ascii=False, indent=1)
pd.set_option('display.width', 250)
for lab in R.diseno.unique():
    print(lab); print(R[(R.diseno == lab) & (R.bloque == '1-3')][['modelo', 'fa_sup', 'fa_sup_lo', 'fa_sup_hi', 'fa_inf', 'fa_bil', 'det10', 'det20', 'det20_lo', 'det20_hi', 'umbral_sup_mediano_pct']].round(1).to_string(index=False))
print(pd.DataFrame(dif).round(1).to_string(index=False)); print(PC.round(1).to_string(index=False)); print(yrs)
print(R[(R.diseno.str.startswith('D1')) & (R.bloque != '1-3')].pivot(index='modelo', columns='anio_prueba', values='fa_sup').round(1))
