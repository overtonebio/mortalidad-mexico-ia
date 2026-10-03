"""v5: semillas con rng=np.random.default_rng(...) (statsmodels 0.15 rechaza random_state y el fallo se ocultaba en el except); los fallos se registran en _fallo_*.
Benchmark a gran escala: 8 modelos x 1,328 series de mortalidad x 3 orígenes (h=36)."""
import os, time, warnings, numpy as np, pandas as pd, torch
from multiprocessing import Pool
warnings.filterwarnings('ignore')
Q = np.array([0.1, 0.5, 0.9]); H = 36
ORIGINS = {'2014-12': 36, '2015-12': 48, '2016-12': 60}     # number of context months (from 2012-01)

def _q(s): return np.percentile(s, [10, 50, 90], axis=0).T

def classical(y, h, seed=0):
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    rng = np.random.default_rng(seed); z = np.log1p(y); n = len(y); out = {}
    pt = np.resize(y[-12:], h); e = y[12:] - y[:-12]; e = e - e.mean()
    out['Ingenuo estacional'] = (pt, _q(np.maximum(pt + rng.choice(e, (500, h)), 0)))
    t = np.arange(n + h) / 12; mo = np.arange(n + h) % 12
    X = sm.add_constant(np.column_stack([t, np.eye(12)[mo][:, 1:]]))
    r = sm.OLS(z, X[:n]).fit(); B = rng.multivariate_normal(r.params, r.cov_params(), 500)
    sims = np.expm1(X[n:] @ B.T + rng.normal(0, np.sqrt(r.scale), (h, 500))).T
    out['Log-lineal'] = (np.expm1(X[n:] @ r.params + r.scale / 2), _q(sims))
    try:
        m = SARIMAX(z, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        s = np.expm1(m.simulate(h, repetitions=500, anchor='end', rng=np.random.default_rng(seed)).squeeze().T); out['SARIMA'] = (np.median(s, 0), _q(s))
    except Exception as e:
        out['SARIMA'] = out['Log-lineal']; out['_fallo_SARIMA'] = repr(e)[:120]
    try:
        m = ETSModel(pd.Series(z), error='add', trend='add', damped_trend=True, seasonal='add', seasonal_periods=12).fit(disp=False)
        s = np.expm1(m.simulate(h, repetitions=500, anchor='end', rng=np.random.default_rng(seed + 1)).values.T); out['ETS amortiguado'] = (np.median(s, 0), _q(s))
    except Exception as e:
        out['ETS amortiguado'] = out['Log-lineal']; out['_fallo_ETS'] = repr(e)[:120]
    return out

def _cls_job(args):
    i, y = args
    return i, classical(y, H, seed=i)

def metrics(y_tr, y_te, pt, q):
    scale = np.mean(np.abs(y_tr[12:] - y_tr[:-12])) + 1e-9
    wql = sum(2 * np.sum(np.maximum(Q[k] * (y_te - q[:, k]), (Q[k] - 1) * (y_te - q[:, k]))) for k in range(3)) / (3 * np.sum(np.abs(y_te)) + 1e-9)
    ann = [abs(y_te[12*k:12*k+12].sum() / max(pt[12*k:12*k+12].sum(), 1e-9) - 1) * 100 for k in range(3)]
    return dict(MASE=np.mean(np.abs(y_te - pt)) / scale, WQL=wql,
                cov80=np.mean((y_te >= q[:, 0]) & (y_te <= q[:, 2])) * 100, err_anual=np.mean(ann))

def foundation(ctxs, h):
    from chronos import BaseChronosPipeline
    import timesfm
    out = {}
    ch2 = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu')
    qs = []
    for b in range(0, len(ctxs), 256):
        q, _ = ch2.predict_quantiles([c[None, :].astype(np.float32) for c in ctxs[b:b+256]], prediction_length=h, quantile_levels=list(Q))
        qs += [x[0].numpy() for x in q]
    out['Chronos-2'] = [(q[:, 1], q) for q in qs]
    chb = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-base', device_map='cpu')
    qs = []
    for b in range(0, len(ctxs), 256):
        q, _ = chb.predict_quantiles(torch.tensor(np.stack(ctxs[b:b+256]), dtype=torch.float32), prediction_length=h, quantile_levels=list(Q))
        qs += list(q.numpy())
    out['Chronos-Bolt'] = [(q[:, 1], q) for q in qs]
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True, per_core_batch_size=64,
                use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    _, qt = tfm.forecast(horizon=h, inputs=[c.astype(float) for c in ctxs])
    out['TimesFM-2.5'] = [(qt[i, :h, 5], qt[i, :h][:, [1, 5, 9]]) for i in range(len(ctxs))]
    return out

ENS = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']

if __name__ == '__main__':
    torch.set_num_threads(10)
    C = pd.read_csv('corpus_mortalidad_mensual.csv.gz', index_col=0)
    Y = C[[str(i) for i in range(156)]].values.astype(float)
    rows = []; t0 = time.time()
    for oname, n in ORIGINS.items():
        ctxs = [Y[i, :n] for i in range(len(Y))]
        fm = foundation(ctxs, H); print(oname, 'foundation', round(time.time() - t0), flush=True)
        with Pool(10) as p:
            cl = dict(p.map(_cls_job, [(i, ctxs[i]) for i in range(len(Y))], chunksize=8))
        print(oname, 'classical', round(time.time() - t0), flush=True)
        for i, sid in enumerate(C.index):
            res = dict(cl[i]); res.update({k: v[i] for k, v in fm.items()})
            P = np.vstack([res[m][0] for m in ENS]); QQ = np.stack([res[m][1] for m in ENS])
            res['Ensamble (mediana)'] = (np.median(P, 0), np.median(QQ, 0))
            y_te = Y[i, n:n + H]
            for m, (pt, q) in res.items():
                rows.append(dict(sid=sid, origin=oname, model=m, **metrics(ctxs[i], y_te, np.maximum(pt, 0), np.maximum(q, 0))))
    R = pd.DataFrame(rows); R.to_csv('benchmark_resultados.csv.gz', index=False)
    print('done', round(time.time() - t0), 's', len(R))
