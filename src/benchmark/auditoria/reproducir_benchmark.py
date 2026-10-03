"""Reproducción consolidada de TODAS las cifras del benchmark (antes repartidas entre analysis2.py y celdas
interactivas 398-451 de la sesión). Entrada: corpus2_mortalidad_mensual.csv.gz y fc_<origen>.npz.
Uso: python reproducir_benchmark.py <dir_fc> <dir_salida>
Salidas (en <dir_salida>): numeros_repro.json, metricas_por_serie.csv.gz, tabla1_repro.csv, atlas_repro.csv.gz,
friedman_por_serie.json. No escribe en ningún otro lugar."""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

FCD, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
C = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(float); NS = len(Y)
Q = np.array([0.1, 0.5, 0.9])
VAL = {'2014-12': 36, '2015-12': 48, '2016-12': 60}
FC = {o: dict(np.load(f'{FCD}/fc_{o}.npz')) for o in list(VAL) + ['2019-12']}
ENS6 = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']
for o in FC: FC[o]['Ensamble (mediana)'] = np.median(np.stack([FC[o][m] for m in ENS6]), 0)
MODELS = list(FC['2016-12'].keys())
TYPE = {'Cánceres': [g for g in C.grp.unique() if g.startswith('C. ') or g in ('Leucemias', 'Linfomas y mieloma', 'Tumores SNC')],
        'Crónicas no transmisibles': ['Cardiopatía isquémica', 'Enf. cerebrovascular', 'Enf. hipertensivas', 'Diabetes mellitus',
                                      'EPOC', 'Enf. hepáticas', 'Insuficiencia renal'],
        'Infecciosas y nutricionales': ['Influenza y neumonía', 'VIH', 'Desnutrición'],
        'Causas externas': ['Homicidio', 'Suicidio', 'Accidentes de tránsito'],
        'Mal definidas / todas': ['Causas mal definidas', 'Todas las causas']}
g2t = {g: t for t, gs in TYPE.items() for g in gs}; C['tipo'] = C.grp.map(g2t)
# volumen: 6 estratos (así se calcularon las cifras publicadas; analysis2.py original usaba 4)
C['vol'] = pd.cut(C.mean_pre, [0, 10, 25, 100, 300, 1000, 1e7], labels=['0-10', '10-25', '25-100', '100-300', '300-1000', '>1000'])
C['vol4'] = pd.cut(C.mean_pre, [0, 10, 25, 100, 1e7], labels=['0-10', '10-25', '25-100', '>100'])

def wql(y, q):
    return sum(2 * np.sum(np.maximum(Q[k] * (y - q[..., k]), (Q[k] - 1) * (y - q[..., k])), axis=-1) for k in range(3)) / (3 * np.sum(np.abs(y), axis=-1) + 1e-9)

rows = []
for o, n in VAL.items():
    yt = Y[:, n:n + 36]; ytr = Y[:, :n]; scale = np.mean(np.abs(ytr[:, 12:] - ytr[:, :-12]), 1) + 1e-9
    for m in MODELS:
        q = FC[o][m][:, :36]; pt = q[..., 1]
        b = pd.DataFrame({'sid': C.index, 'origin': o, 'model': m, 'MASE': np.mean(np.abs(yt - pt), 1) / scale, 'WQL': wql(yt, q),
                          'cov80': np.mean((yt >= q[..., 0]) & (yt <= q[..., 2]), 1) * 100})
        for k in range(3):
            sl = slice(12 * k, 12 * k + 12); b[f'WQL_h{k+1}'] = wql(yt[:, sl], q[:, sl])
            b[f'logerr_y{k+1}'] = np.log((yt[:, sl].sum(1) + 1) / (pt[:, sl].sum(1) + 1))
        rows.append(b)
M = pd.concat(rows, ignore_index=True).merge(C[['grp', 'tipo', 'level', 'vol', 'vol4', 'mean_pre']], left_on='sid', right_index=True)
M.round(6).to_csv(f'{OUT}/metricas_por_serie.csv.gz', index=False)

QA = {8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219}
def rank_table(P):
    rk = P.rank(axis=1).mean().sort_values(); k = P.shape[1]; N = len(P)
    fr = stats.friedmanchisquare(*[P[c] for c in P.columns])
    return rk, QA[k] * np.sqrt(k * (k + 1) / (6 * N)), fr.statistic, fr.pvalue, N
def rel_vs(df, metric, ref='Log-lineal'):
    P = df.pivot_table(index=['sid', 'origin'], columns='model', values=metric)
    return np.exp(np.log(P.div(P[ref], axis=0)).mean())

P_eval = M.pivot_table(index=['sid', 'origin'], columns='model', values='WQL')
rk_all, cd_all, fchi, fp, N_all = rank_table(P_eval)                    # publicado: bloques = serie × origen
P_ser = P_eval.groupby(level=0).mean()                                     # B: bloques = serie (WQL medio de 3 orígenes)
rk_s, cd_s, fchi_s, fp_s, N_s = rank_table(P_ser)
rk_s2 = P_eval.rank(axis=1).groupby(level=0).mean()                        # alternativa: rango medio por serie
fr_s2 = stats.friedmanchisquare(*[rk_s2[c] for c in rk_s2.columns])
# Nemenyi: pares no distinguibles (|Δrango| < DC) en la versión por serie
order = rk_s.index.tolist(); nd = [(a, b, round(abs(rk_s[a] - rk_s[b]), 3)) for i, a in enumerate(order) for b in order[i + 1:] if abs(rk_s[a] - rk_s[b]) < cd_s]
json.dump(dict(publicado=dict(N=int(N_all), CD=float(cd_all), chi2=float(fchi), p=float(fp), rangos=rk_all.round(3).to_dict()),
               por_serie_WQLmedio=dict(N=int(N_s), CD=float(cd_s), chi2=float(fchi_s), p=float(fp_s), rangos=rk_s.round(3).to_dict(),
                                       pares_no_distinguibles=nd),
               por_serie_rangomedio=dict(N=int(len(rk_s2)), chi2=float(fr_s2.statistic), p=float(fr_s2.pvalue),
                                         rangos=rk_s2.mean().sort_values().round(3).to_dict())),
          open(f'{OUT}/friedman_por_serie.json', 'w'), indent=1, ensure_ascii=False)

rel_all = rel_vs(M, 'WQL'); cov = M.groupby('model').cov80.mean(); cov_o = M.groupby(['model', 'origin']).cov80.mean().unstack()
hor = pd.DataFrame({f'h{b}': rel_vs(M, f'WQL_h{b}') for b in (1, 2, 3)})
by_tipo = pd.DataFrame({t: rel_vs(M[M.tipo == t], 'WQL') for t in TYPE})
by_level = pd.DataFrame({l: rel_vs(M[M.level == l], 'WQL') for l in ['Nacional', 'Nacional×edad', 'Estatal', 'Estatal×edad']})
L = np.log(P_eval.div(P_eval['Log-lineal'], axis=0)); gidx = C.grp.reindex(L.index.get_level_values(0)).values
G = L.groupby(gidx); gs = list(G.groups); Ssum = G.sum(); Scnt = G.count(); rng = np.random.default_rng(1); boot = []
for _ in range(2000):
    s = rng.choice(len(gs), len(gs)); boot.append(np.exp(Ssum.iloc[s].sum() / Scnt.iloc[s].sum()))
boot = pd.DataFrame(boot); ci = pd.DataFrame({'rel': rel_all, 'lo': boot.quantile(0.025), 'hi': boot.quantile(0.975)})
# meta-regresión
pre = Y[:, :96]; t = np.arange(96); lg = np.log1p(pre)
X = np.column_stack([np.ones(96), t / 12] + [np.sin(2 * np.pi * t * k / 12) for k in (1, 2)] + [np.cos(2 * np.pi * t * k / 12) for k in (1, 2)])
beta, *_ = np.linalg.lstsq(X, lg.T, rcond=None); res_ = lg.T - X @ beta
feat = pd.DataFrame({'log_vol': np.log(C.mean_pre.values), 'ruido': res_.std(0), 'tendencia': beta[1],
                     'estacionalidad': np.sqrt(beta[2] ** 2 + beta[3] ** 2)}, index=C.index)
fm = M[M.model.isin(['Chronos-2', 'TimesFM-2.5', 'Chronos-Bolt'])].groupby(['sid', 'origin']).WQL.mean()
ll = M[M.model == 'Log-lineal'].set_index(['sid', 'origin']).WQL
mr = pd.DataFrame({'y': np.log(fm / ll)}).reset_index().merge(feat, left_on='sid', right_index=True); mr['grp'] = C.grp.reindex(mr.sid).values
for c in ['log_vol', 'ruido', 'tendencia', 'estacionalidad']: mr[c + '_z'] = (mr[c] - mr[c].mean()) / mr[c].std()
meta = smf.ols('y ~ log_vol_z + ruido_z + tendencia_z + estacionalidad_z', mr).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(mr.grp)[0]})
# conformal placebo (celda 435): calibración 2014+2015, prueba 2016
def conformal_rows(cal, tst, strata, alpha=0.9):
    out = []
    for b in (1, 2, 3):
        c = cal.copy(); c['s'] = np.abs(c[f'logerr_y{b}'])
        qt = c.groupby(['model'] + strata, observed=True).s.apply(lambda e: np.quantile(e, min(1, np.ceil((len(e) + 1) * alpha) / len(e)), method='higher')).rename('q').reset_index()
        tt = tst.merge(qt, on=['model'] + strata, how='left'); tt['half'] = (np.exp(tt.q) - 1) * 100
        tt['flag'] = np.abs(tt[f'logerr_y{b}']) > tt.q; tt['block'] = b; out.append(tt[['sid', 'model', 'level', 'block', 'half', 'flag']])
    return pd.concat(out)
cal = M[M.origin.isin(['2014-12', '2015-12'])]; tst = M[M.origin == '2016-12']; PL = {}
for lab, strata in [('Mondrian volumen', ['vol']), ('Mondrian volumen×nivel', ['vol', 'level'])]:
    r = conformal_rows(cal, tst, strata); g = r.groupby('model'); gn = r[r.level == 'Nacional'].groupby('model')
    PL[lab] = pd.DataFrame({'FPR': g.flag.mean() * 100, 'semiancho_mediana': g.half.median(), 'FPR_nacional': gn.flag.mean() * 100,
                            'semiancho_nacional': gn.half.median()})
# contrafactual 2020-2023 (origen 2019-12), conformal Mondrian volumen con los 3 orígenes (año 4 usa bloque 3)
qtab2 = {}
for (m, v), g in M.groupby(['model', 'vol'], observed=True):
    for b in (1, 2, 3):
        e = np.abs(g[f'logerr_y{b}'].values); nn = len(e); qtab2[(m, v, b)] = np.quantile(e, min(1, np.ceil((nn + 1) * 0.9) / nn), method='higher')
cf = []
for m in MODELS:
    pt = FC['2019-12'][m][..., 1]
    for k, yr in enumerate(range(2020, 2024)):
        sl = slice(96 + 12 * k, 108 + 12 * k); o_ = Y[:, sl].sum(1); e_ = pt[:, 12 * k:12 * k + 12].sum(1); b = min(k + 1, 3)
        qq = np.array([qtab2[(m, v, b)] for v in C.vol])
        cf.append(pd.DataFrame({'sid': C.index, 'model': m, 'year': yr, 'obs': o_, 'exp': e_, 'pscore': (o_ / e_ - 1) * 100,
                                'lo': (o_ / (e_ * np.exp(qq)) - 1) * 100, 'hi': (o_ / (e_ * np.exp(-qq)) - 1) * 100}))
CF = pd.concat(cf, ignore_index=True).merge(C[['grp', 'sex', 'geo', 'ab', 'level', 'tipo', 'vol']], left_on='sid', right_index=True)
CF['signif'] = np.where(CF.lo > 0, 'exceso', np.where(CF.hi < 0, 'déficit', 'ns'))
CF.round(3).to_csv(f'{OUT}/atlas_repro.csv.gz', index=False)
nat = CF[CF.level == 'Nacional']; ens = 'Ensamble (mediana)'
pos = ['Diabetes mellitus', 'Cardiopatía isquémica', 'Influenza y neumonía', 'Todas las causas', 'Enf. hipertensivas']
posall = CF[CF.grp.isin(['Diabetes mellitus', 'Cardiopatía isquémica', 'Todas las causas']) & CF.year.isin([2020, 2021]) & (CF.ab == 'Todas')]
a22 = nat[(nat.tipo == 'Cánceres') & (nat.model == ens) & (nat.year == 2022)]
cn = C[(C.level == 'Nacional') & (C.tipo == 'Cánceres')].index; pl = M[(M.model == ens) & M.sid.isin(cn)]
plac_vals = pd.DataFrame([dict(origin=o, block=b, mean_pct=(np.exp(pl[pl.origin == o][f'logerr_y{b}'].mean()) - 1) * 100) for o in VAL for b in (1, 2, 3)])
real = CF[(CF.model == ens) & CF.sid.isin(cn)]
rv = real.groupby('year').apply(lambda d: pd.Series({'mean_pct': (np.exp(np.log(d.obs / d.exp).mean()) - 1) * 100, 'frac_neg': (d.pscore < 0).mean() * 100}))
N2 = dict(n_series=int(NS), levels=C.level.value_counts().to_dict(), n_groups=int(C.grp.nunique()), median_vol=round(float(C.mean_pre.median()), 1),
          n_eval=int(N_all), CD=round(float(cd_all), 2), rank=rk_all.round(2).to_dict(), rel=ci.round(3).to_dict('index'), cov=cov.round(1).to_dict(),
          cov_por_origen=cov_o.round(1).to_dict('index'), hor=hor.round(3).to_dict('index'), by_level=by_level.round(3).to_dict('index'),
          by_tipo=by_tipo.round(3).to_dict('index'),
          meta=pd.DataFrame({'coef': meta.params, 'p': meta.pvalues}).round(4).to_dict('index'),
          plac_vol=PL['Mondrian volumen'].round(1).to_dict('index'), plac_volniv=PL['Mondrian volumen×nivel'].round(1).to_dict('index'),
          posctrl_2020=nat[nat.grp.isin(pos) & (nat.year == 2020) & (nat.model == ens)][['grp', 'sex', 'pscore', 'lo', 'hi']].round(1).to_dict('records'),
          det_pos=posall.groupby('model').signif.apply(lambda s: (s == 'exceso').mean() * 100).round(1).to_dict(), n_pos_series=int(posall.sid.nunique()),
          atlas22=a22[['grp', 'sex', 'pscore', 'lo', 'hi']].round(1).to_dict('records'), atlas_neg22=int((a22.pscore < 0).sum()), atlas_n=int(len(a22)),
          joint_placebo=plac_vals.round(2).to_dict('records'), joint_real={str(k): v for k, v in rv.round(1).to_dict('index').items()})
json.dump(N2, open(f'{OUT}/numeros_repro.json', 'w'), indent=1, default=str, ensure_ascii=False)
fam = {'LightGBM global': 'Global (IA)', 'N-HiTS global': 'Global (IA)', 'Chronos-2': 'Fundacional (IA)', 'Chronos-Bolt': 'Fundacional (IA)',
       'TimesFM-2.5': 'Fundacional (IA)', 'Ensamble (mediana)': 'Ensamble', 'Log-lineal': 'Clásico', 'ETS amortiguado': 'Clásico', 'SARIMA': 'Clásico',
       'Ingenuo estacional': 'Clásico'}
PV = PL['Mondrian volumen']
pd.DataFrame([{'Modelo': m, 'Familia': fam[m], 'Rango WQL': round(rk_all[m], 2), 'Rango WQL (por serie)': round(rk_s[m], 2),
               'WQL_rel': round(ci.loc[m, 'rel'], 3), 'IC_lo': round(ci.loc[m, 'lo'], 3), 'IC_hi': round(ci.loc[m, 'hi'], 3),
               'Cobertura80': round(cov[m], 1), 'FPR': round(PV.loc[m, 'FPR'], 1), 'Semiancho': round(PV.loc[m, 'semiancho_mediana'], 1)}
              for m in rk_all.index]).to_csv(f'{OUT}/tabla1_repro.csv', index=False)
print('ok', N_all, round(cd_all, 3), N_s, round(cd_s, 3))
