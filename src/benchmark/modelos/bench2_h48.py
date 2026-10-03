"""Validación a 48 meses (auditoría): mismos modelos y configuración que bench2.py, horizonte 48.
Orígenes dic-2014 (36 meses de contexto; prueba 2015-2018) y dic-2015 (48; prueba 2016-2019).
Salida: fc_h48_<origen>.npz. No modifica resultados originales."""
import os, sys, time, numpy as np, torch
sys.path.append(os.getcwd())
import bench as B, bench2 as B2
from multiprocessing import Pool
ORIG = {'2014-12': 36, '2015-12': 48}; H = 48
if __name__ == '__main__':
    torch.set_num_threads(10); t0 = time.time()
    for o, n in ORIG.items():
        F = {}; ctxs = [B2.Y[i, :n] for i in range(B2.NS)]
        fm = B.foundation(ctxs, H)
        for k, v in fm.items(): F[k] = np.stack([q for _, q in v])
        print(o, 'foundation', round(time.time() - t0), flush=True)
        with Pool(10) as p: cl = dict(p.map(B2._cls, [(i, ctxs[i], H) for i in range(B2.NS)], chunksize=8))
        for m in ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado']:
            F[m] = np.stack([cl[i][m][1] for i in range(B2.NS)])
        print(o, 'classical', round(time.time() - t0), flush=True)
        F['LightGBM global'], _ = B2.lgbm_global(n, H); print(o, 'lgbm', round(time.time() - t0), flush=True)
        F['N-HiTS global'], _ = B2.nhits_global(n, H); print(o, 'nhits', round(time.time() - t0), flush=True)
        np.savez_compressed(f'fc_h48_{o}.npz', **{k: np.maximum(v, 0).astype(np.float32) for k, v in F.items()})
    print('done', round(time.time() - t0))
