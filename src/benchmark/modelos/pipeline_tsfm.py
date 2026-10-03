"""Counterfactual forecasting of breast-cancer mortality (women >=20, Mexico).
Classical (seasonal naive, log-linear trend+month, SARIMA, damped ETS) vs zero-shot
time-series foundation models (Chronos-2, Chronos-Bolt-base, TimesFM-2.5) + median ensemble.
Rolling-origin backtest (9 quarterly origins 2014-12..2016-12, h=36) -> split-conformal
intervals for 12-month blocks; counterfactual from origin 2019-12, h=60."""
import numpy as np, pandas as pd, torch, warnings, json, time
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from chronos import BaseChronosPipeline
import timesfm
warnings.filterwarnings('ignore'); torch.set_num_threads(8); torch.manual_seed(0)
rng = np.random.default_rng(2026)
Q = [0.1, 0.5, 0.9]

S = pd.read_csv('series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
import os
TARGETS = os.environ.get('TARGETS', 'asmr,deaths_adj,asmr_20-39,asmr_40-59,asmr_60+').split(',')
SUF = os.environ.get('SUF', '')

ch2 = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu')
chb = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-base', device_map='cpu')
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True))

def _q(sims):  # sims (nsim,h) -> (h,3)
    return np.percentile(sims, [10, 50, 90], axis=0).T

def m_snaive(y, h):
    pt = np.resize(y[-12:], h); e = y[12:] - y[:-12]; e = e - e.mean()
    sims = pt + rng.choice(e, (2000, h)); return pt, _q(sims)

def m_loglin(y, h):
    n = len(y); t = np.arange(n + h) / 12; mo = np.arange(n + h) % 12
    X = sm.add_constant(np.column_stack([t, np.eye(12)[mo][:, 1:]]))
    r = sm.OLS(np.log(y), X[:n]).fit()
    B = rng.multivariate_normal(r.params, r.cov_params(), 2000)
    sims = np.exp(X[n:] @ B.T + rng.normal(0, np.sqrt(r.scale), (h, 2000))).T
    return np.exp(X[n:] @ r.params + r.scale / 2), _q(sims)

def m_sarima(y, h):
    m = SARIMAX(np.log(y), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    sims = np.exp(m.simulate(h, repetitions=2000, anchor='end', rng=np.random.default_rng(int(rng.integers(1 << 31)))).squeeze().T)
    return sims.mean(0), _q(sims)

def m_ets(y, h):
    m = ETSModel(pd.Series(np.log(y)), error='add', trend='add', damped_trend=True, seasonal='add',
                 seasonal_periods=12).fit(disp=False)
    sims = np.exp(m.simulate(h, repetitions=2000, anchor='end', rng=np.random.default_rng(int(rng.integers(1 << 31)))).values.T)
    return sims.mean(0), _q(sims)

def fm_batch(ys, h):
    """zero-shot foundation models on a list of context arrays -> dict model -> list[(pt,q)]"""
    out = {}
    q, mean = ch2.predict_quantiles([y[None, :].astype(np.float32) for y in ys], prediction_length=h,
                                    quantile_levels=Q)
    out['Chronos-2'] = [(q[i][0, :, 1].numpy(), q[i][0].numpy()) for i in range(len(ys))]
    L = max(len(y) for y in ys)
    ctx = [torch.tensor(y, dtype=torch.float32) for y in ys]
    qb, _ = chb.predict_quantiles(ctx, prediction_length=h, quantile_levels=Q)
    out['Chronos-Bolt'] = [(qb[i, :, 1].numpy(), qb[i].numpy()) for i in range(len(ys))]
    pt, qt = tfm.forecast(horizon=h, inputs=[y.astype(float) for y in ys])
    # timesfm quantile head: [mean, q0.1..q0.9]
    out['TimesFM-2.5'] = [(qt[i, :h, 5], qt[i, :h, [1, 5, 9]].T) for i in range(len(ys))]
    return out

CLASSIC = {'Naive estacional': m_snaive, 'Log-lineal (tendencia+mes)': m_loglin,
           'SARIMA': m_sarima, 'ETS amortiguado': m_ets}
ENS_MEMBERS = ['Log-lineal (tendencia+mes)', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']

def run_origin(origin, h):
    ctxs = {tg: S[tg][:origin].values.astype(float) for tg in TARGETS}
    res = {tg: {} for tg in TARGETS}
    for tg, y in ctxs.items():
        for nm, f in CLASSIC.items():
            res[tg][nm] = f(y, h)
    fm = fm_batch([ctxs[tg] for tg in TARGETS], h)
    for nm, lst in fm.items():
        for i, tg in enumerate(TARGETS):
            res[tg][nm] = lst[i]
    for tg in TARGETS:
        P = np.vstack([res[tg][m][0] for m in ENS_MEMBERS]); Qs = np.stack([res[tg][m][1] for m in ENS_MEMBERS])
        res[tg]['Ensamble (mediana)'] = (np.median(P, 0), np.median(Qs, 0))
    return res

# ---------------- backtest ----------------
t0 = time.time()
origins = pd.date_range('2014-12-01', '2016-12-01', freq='3MS')
H = 36
mrows, brows = [], []
for o in origins:
    res = run_origin(o, H)
    te_idx = pd.date_range(o + pd.DateOffset(months=1), periods=H, freq='MS')
    for tg in TARGETS:
        y_tr = S[tg][:o].values; y_te = S[tg][te_idx].values
        scale = np.mean(np.abs(y_tr[12:] - y_tr[:-12]))
        for nm, (pt, q) in res[tg].items():
            mrows.append(dict(origin=o.strftime('%Y-%m'), target=tg, model=nm,
                              MASE=np.mean(np.abs(y_te - pt)) / scale,
                              sMAPE=np.mean(2 * np.abs(y_te - pt) / (np.abs(y_te) + np.abs(pt))) * 100,
                              cov80=np.mean((y_te >= q[:, 0]) & (y_te <= q[:, 2])) * 100))
            for k in range(3):
                sl = slice(12 * k, 12 * k + 12)
                brows.append(dict(origin=o.strftime('%Y-%m'), target=tg, model=nm, block=k + 1,
                                  logerr=np.log(y_te[sl].sum() / pt[sl].sum())))
    print('origin', o.strftime('%Y-%m'), round(time.time() - t0), 's', flush=True)
bm = pd.DataFrame(mrows); bb = pd.DataFrame(brows)
bm.to_csv(f'tsfm_backtest_mensual{SUF}.csv', index=False); bb.to_csv(f'tsfm_backtest_bloques{SUF}.csv', index=False)

# ---------------- counterfactual ----------------
o = pd.Timestamp('2019-12-01'); HC = 60
res = run_origin(o, HC)
hz = pd.date_range('2020-01-01', periods=HC, freq='MS')
fc_rows, ex_rows = [], []
n_cal = bb.origin.nunique()
qlev = min(1.0, np.ceil((n_cal + 1) * 0.9) / n_cal)          # split-conformal 90%
for tg in TARGETS:
    obs = S[tg][hz].values
    for nm, (pt, q) in res[tg].items():
        for i, dt in enumerate(hz):
            fc_rows.append(dict(target=tg, model=nm, fecha=dt, obs=obs[i], pred=pt[i], q10=q[i, 0], q90=q[i, 2]))
        for k, yr in enumerate(range(2020, 2025)):
            sl = slice(12 * k, 12 * k + 12)
            kb = min(k + 1, 3)
            r = np.abs(bb[(bb.target == tg) & (bb.model == nm) & (bb.block == kb)].logerr.values)
            c = np.quantile(r, qlev, method='higher')
            o_, p_ = obs[sl].sum(), pt[sl].sum()
            ex_rows.append(dict(target=tg, model=nm, year=yr, validated=k < 3, observed=o_, expected=p_,
                                pscore=(o_ / p_ - 1) * 100,
                                p_lo=(o_ / (p_ * np.exp(c)) - 1) * 100, p_hi=(o_ / (p_ * np.exp(-c)) - 1) * 100))
pd.DataFrame(fc_rows).to_csv(f'tsfm_contrafactual_mensual{SUF}.csv', index=False)
pd.DataFrame(ex_rows).to_csv(f'tsfm_exceso_anual{SUF}.csv', index=False)
json.dump(dict(origins=[x.strftime('%Y-%m') for x in origins], H=H, conformal_level=0.9, n_cal=int(n_cal),
               q_used=float(qlev), members=ENS_MEMBERS, runtime_s=round(time.time() - t0)),
          open(f'tsfm_meta{SUF}.json', 'w'), indent=1)
print('done', round(time.time() - t0), 's')
