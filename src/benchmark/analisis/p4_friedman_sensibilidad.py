"""P4 (v5). Sensibilidad de Friedman/Nemenyi a la dependencia entre series anidadas.
Base (v4): una observación por serie = WQL medio de los 3 orígenes, N = 2 870.
Sensibilidades:
 (1) Conglomerado por grupo de causa (N = 28): rango medio de cada modelo dentro del grupo → Friedman sobre 28 bloques.
 (2) Conglomerado por causa × sexo (N = 52, las series nacionales definen los conglomerados).
 (3) Subconjunto sin anidamiento de conteos: series estatales de todas las edades excluyendo 'Todas las causas'
     (grupos CIE-10 disjuntos, entidades disjuntas).
 (4) Por nivel de agregación (nacional, nacional×edad, estatal, estatal×edad).
Nemenyi: CD = q_0.05(k) · sqrt(k(k+1)/(6N)), q_0.05(10) = 3.164.
Además: IC 95 % bootstrap por grupo de causa (2 000 réplicas) de la diferencia de rango medio entre pares clave."""
import numpy as np, pandas as pd, json
from scipy import stats
M = pd.read_csv('resultados/benchmark/metricas_por_serie.csv.gz')
C = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)[['grp', 'sex', 'level', 'ab']]
W = M.groupby(['sid', 'model']).WQL.mean().unstack(); models = list(W.columns); k = len(models)
R = W.rank(axis=1); R = R.join(C)
qa = 3.164
def fried(ranks_df):
    N = len(ranks_df); rm = ranks_df[models].mean()
    chi = 12 * N / (k * (k + 1)) * (np.sum(rm ** 2) - k * (k + 1) ** 2 / 4)
    return dict(N=N, chi2=float(chi), p=float(stats.chi2.sf(chi, k - 1)), CD=float(qa * np.sqrt(k * (k + 1) / (6 * N))), rangos=rm.round(3).to_dict())
out = {}
out['base_por_serie'] = fried(R)
g = R.groupby('grp')[models].mean().rank(axis=1)   # re-rango dentro de conglomerado
out['conglomerado_grupo_N28_rango_de_rangos_medios'] = fried(g)
g2 = R.groupby('grp')[models].mean()
out['conglomerado_grupo_N28_rango_medio_sin_rerango'] = dict(rangos=g2.mean().round(3).to_dict(), N=len(g2))
R['cs'] = R.grp + '|' + R.sex.astype(str)
g3 = R.groupby('cs')[models].mean().rank(axis=1); out['conglomerado_causa_sexo'] = fried(g3)
sub = R[(R.level == 'Estatal') & (R.grp != 'Todas las causas')]; out['estatal_sin_todas_las_causas'] = fried(sub)
for lv in ['Nacional', 'Nacional×edad', 'Estatal', 'Estatal×edad']: out[f'nivel_{lv}'] = fried(R[R.level == lv])
# bootstrap por grupo de la diferencia de rango medio (por serie) entre pares
pairs = [('LightGBM global', 'Ensamble (mediana)'), ('Ensamble (mediana)', 'TimesFM-2.5'), ('TimesFM-2.5', 'Chronos-2'),
         ('Chronos-2', 'Chronos-Bolt'), ('Chronos-2', 'N-HiTS global'), ('N-HiTS global', 'Log-lineal'), ('Log-lineal', 'ETS amortiguado')]
rng = np.random.default_rng(2026); G = R.grp.unique(); gi = {x: np.flatnonzero(R.grp.values == x) for x in G}
RV = R[models].values; bs = []
for _ in range(2000):
    s = np.concatenate([gi[x] for x in rng.choice(G, len(G))]); bs.append(RV[s].mean(0))
bs = np.array(bs); pt = RV.mean(0); mi = {m: i for i, m in enumerate(models)}
out['pares_bootstrap_grupo'] = [dict(a=a, b=b, dif=float(pt[mi[a]] - pt[mi[b]]), lo=float(np.percentile(bs[:, mi[a]] - bs[:, mi[b]], 2.5)),
                                     hi=float(np.percentile(bs[:, mi[a]] - bs[:, mi[b]], 97.5))) for a, b in pairs]
json.dump(out, open('resultados/p4_friedman_sensibilidad.json', 'w'), ensure_ascii=False, indent=1)
rows = []
for key, v in out.items():
    if 'rangos' in v and 'CD' in v: rows.append(dict(analisis=key, N=v['N'], chi2=round(v['chi2'], 1), p=v['p'], CD=round(v['CD'], 3), **{m: round(v['rangos'][m], 2) for m in models}))
T = pd.DataFrame(rows); T.to_csv('resultados/p4_friedman_sensibilidad.csv', index=False)
pd.set_option('display.width', 300); print(T.to_string()); print(pd.DataFrame(out['pares_bootstrap_grupo']).round(3).to_string())
