"""Clasificador de subregistro: P(causa básica = C50 | características sociodemográficas y de la defunción)
entrenado en defunciones bien definidas de mujeres >=20 años (2012-2018), validado en 2019, y aplicado
a defunciones con causas poco útiles (R00-R99 y C76-C80) para estimar muertes por cáncer de mama 'ocultas'."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

W = pd.read_csv('mujeres20_individual_2012_2024.csv.gz')
W = W[W.c3.notna()]
W['is_R'] = W.c3.str.startswith('R'); W['is_C7680'] = W.c3.isin(['C76', 'C77', 'C78', 'C79', 'C80'])
W['is_U07'] = W.c3.str.startswith('U07')
W['garbage'] = W.is_R | W.is_C7680
W['y_c50'] = (W.c3 == 'C50').astype(int)
# harmonised features (codes stable across years)
W['dh_g'] = np.select([W.dh.isin([2, 3, 4, 5, 6, 10]), W.dh.isin([1, 7, 9, 12]), W.dh == 8], [1, 2, 3], 9)
W['sitio_g'] = np.select([W.sitio == 11, W.sitio.isin([1, 2, 3, 4, 5, 6, 7, 8]), W.sitio == 9], [1, 2, 3], 9)
W['tloc_g'] = W.tloc.where(W.tloc.between(1, 17), 99)
W['esc_g'] = W.esc.where(W.esc.between(1, 10), 99)
W['civ_g'] = W.civ.where(W.civ.between(1, 7), 9)
W['asist_g'] = W.asist.where(W.asist.isin([1, 2]), 9)
W['ent_g'] = W.ent.where(W.ent.between(1, 32), 99)
FEAT = ['age', 'ent_g', 'tloc_g', 'esc_g', 'civ_g', 'dh_g', 'sitio_g', 'asist_g', 'm']
CAT = ['ent_g', 'tloc_g', 'esc_g', 'civ_g', 'dh_g', 'sitio_g', 'asist_g']
for c in CAT: W[c] = W[c].astype(int).astype('category')

wd = ~W.garbage & ~W.is_U07
tr = W[wd & W.y.between(2012, 2018)]; va = W[wd & (W.y == 2019)]
params = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=1, num_threads=10)
dtr = lgb.Dataset(tr[FEAT], tr.y_c50, categorical_feature=CAT); dva = lgb.Dataset(va[FEAT], va.y_c50, reference=dtr)
mdl = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
pva = mdl.predict(va[FEAT], num_iteration=mdl.best_iteration)
# age-only logistic benchmark
import statsmodels.api as sm
Xa = lambda d: sm.add_constant(np.column_stack([d.age, d.age ** 2, d.age ** 3]) / np.array([100, 1e4, 1e6]))
lr = sm.Logit(tr.y_c50.values, Xa(tr)).fit(disp=0); pva_age = lr.predict(Xa(va))
perf = dict(n_train=len(tr), n_valid=len(va), prev_valid=va.y_c50.mean(), best_iter=mdl.best_iteration,
            AUC=roc_auc_score(va.y_c50, pva), AUPRC=average_precision_score(va.y_c50, pva), Brier=brier_score_loss(va.y_c50, pva),
            AUC_edad=roc_auc_score(va.y_c50, pva_age), AUPRC_edad=average_precision_score(va.y_c50, pva_age),
            calib_ratio=pva.sum() / va.y_c50.sum())
# calibration by decile
va = va.assign(p=pva); cal = va.groupby(pd.qcut(va.p, 10, duplicates='drop'), observed=True).agg(pred=('p', 'mean'), obs=('y_c50', 'mean')).reset_index(drop=True)
# SHAP (TreeSHAP via pred_contrib) on a sample of validation
smp = va.sample(20000, random_state=1)
contrib = mdl.predict(smp[FEAT], num_iteration=mdl.best_iteration, pred_contrib=True)
shap_imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=FEAT).sort_values(ascending=False)
# application: expected hidden C50 among garbage-coded deaths by year and type
G = W[W.garbage].copy(); G['p'] = mdl.predict(G[FEAT], num_iteration=mdl.best_iteration)
G['tipo'] = np.where(G.is_R, 'R00-R99', 'C76-C80')
hid = G.groupby(['y', 'tipo']).agg(n=('p', 'size'), c50_ocultas=('p', 'sum')).reset_index()
tot = G.groupby('y').agg(n=('p', 'size'), c50_ocultas=('p', 'sum'))
base = tot.loc[2017:2019].mean()
tot['exceso_vs_2017_19'] = tot.c50_ocultas - base.c50_ocultas
# home deaths with garbage codes
hid_home = G[G.sitio_g == 1].groupby('y').p.sum()
