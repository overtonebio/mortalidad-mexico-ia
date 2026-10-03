"""Benchmark v2: 2,870 series; 7 modelos locales/fundacionales + 2 globales entrenados en el corpus
(LightGBM cuantílico directo y N-HiTS). Guarda todos los pronósticos (cuantiles 0.1/0.5/0.9) para análisis post hoc.
Orígenes de validación dic-2014/2015/2016 (h=36) y origen contrafactual dic-2019 (h=60)."""
import os, sys, time, warnings, numpy as np, pandas as pd, torch
from multiprocessing import Pool
sys.path.append(os.getcwd()); warnings.filterwarnings('ignore')
import bench as B
ORIGINS = {'2014-12': (36, 36), '2015-12': (48, 36), '2016-12': (60, 36), '2019-12': (96, 60)}
C = pd.read_csv('corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(float)
NS = len(Y)

def _cls(args):
    i, y, h = args; return i, B.classical(y, h, seed=i)

def lgbm_global(n, h, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    cat = pd.DataFrame({'grp': C.grp.astype('category').cat.codes.values, 'lvl': C.level.astype('category').cat.codes.values,
                        'sex': C.sex.values, 'ab': C.ab.astype('category').cat.codes.values})
    def feats(t, k, idx):
        lags = Y[idx][:, t - 12:t]; s = lags.mean(1) + 1.0
        X = np.column_stack([lags / s[:, None], np.log(s), np.full(len(idx), k), np.full(len(idx), (t + k - 1) % 12),
                             (lags[:, -12:].sum(1) + 1) / (Y[idx][:, t - 24:t - 12].sum(1) + 1) if t >= 24 else np.ones(len(idx)),
                             cat.values[idx]])
        return X, s
    rows, ys = [], []
    allidx = np.arange(NS); frac = min(1.0, 3e6 / (NS * sum(max(0, n - k - 11) for k in range(1, h + 1)) + 1))
    for k in range(1, h + 1):
        for t in range(12, n - k + 1):
            idx = allidx if frac >= 1 else allidx[rng.random(NS) < frac]
            if len(idx) == 0: continue
            X, s = feats(t, k, idx); rows.append(X); ys.append(Y[idx, t + k - 1] / s)
    X = np.vstack(rows).astype(np.float32); yv = np.concatenate(ys).astype(np.float32)
    ncat = X.shape[1] - 4
    catcols = list(range(ncat, X.shape[1]))
    out = np.zeros((NS, h, 3))
    for j, a in enumerate([0.1, 0.5, 0.9]):
        m = lgb.train(dict(objective='quantile', alpha=a, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                           feature_fraction=0.9, verbose=-1, num_threads=10, seed=seed),
                      lgb.Dataset(X, yv, categorical_feature=catcols), num_boost_round=400)
        for k in range(1, h + 1):
            kk = min(k, max(1, n - 12))                       # clamp horizon to trained range if history is short
            Xp, s = feats(n, kk, allidx); Xp[:, 13] = kk; Xp[:, 14] = (n + k - 1) % 12
            out[:, k - 1, j] = m.predict(Xp.astype(np.float32)) * s
    out.sort(axis=2)
    return out, len(yv)

def nhits_global(n, h):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss
    ds = pd.date_range('2012-01-01', periods=n, freq='MS')
    df = pd.DataFrame({'unique_id': np.repeat(np.arange(NS), n), 'ds': np.tile(ds, NS), 'y': Y[:, :n].ravel()})
    m = NHITS(h=h, input_size=24 if n >= 60 else 12, loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), scaler_type='robust',
              max_steps=1000, batch_size=64, windows_batch_size=256, start_padding_enabled=True, random_seed=1,
              enable_progress_bar=False, enable_model_summary=False, logger=False)
    nf = NeuralForecast(models=[m], freq='MS'); nf.fit(df)
    p = nf.predict().reset_index()
    cols = [c for c in p.columns if c.startswith('NHITS')]
    ql = sorted(cols, key=lambda c: (0 if '-lo' in c or '0.1' in c else 2 if '-hi' in c or '0.9' in c else 1))
    arr = np.zeros((NS, h, 3))
    for uid, g in p.groupby('unique_id'):
        arr[int(uid)] = g.sort_values('ds')[ql].values
    arr.sort(axis=2)
    return arr, cols

if __name__ == '__main__':
    torch.set_num_threads(10); t0 = time.time(); only = os.environ.get('ONLY')
    for oname, (n, h) in ORIGINS.items():
        if only and oname != only: continue
        F = {}
        ctxs = [Y[i, :n] for i in range(NS)]
        fm = B.foundation(ctxs, h)
        for k, v in fm.items(): F[k] = np.stack([q for _, q in v])
        print(oname, 'foundation', round(time.time() - t0), flush=True)
        with Pool(10) as p: cl = dict(p.map(_cls, [(i, ctxs[i], h) for i in range(NS)], chunksize=8))
        for mname in ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado']:
            F[mname] = np.stack([cl[i][mname][1] for i in range(NS)])
        print(oname, 'classical', round(time.time() - t0), flush=True)
        F['LightGBM global'], nrow = lgbm_global(n, h); print(oname, 'lgbm', nrow, round(time.time() - t0), flush=True)
        F['N-HiTS global'], cols = nhits_global(n, h); print(oname, 'nhits', cols, round(time.time() - t0), flush=True)
        np.savez_compressed(f'fc_{oname}.npz', **{k: np.maximum(v, 0).astype(np.float32) for k, v in F.items()})
    print('done', round(time.time() - t0))
