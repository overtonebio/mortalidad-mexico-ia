"""v5: recalcula solo los cuatro modelos clásicos con semillas válidas y reemplaza SARIMA/ETS en fc_<origen>.npz y fc_h48_<origen>.npz
de la nueva ejecución (los demás modelos ya coinciden bit a bit con los publicados). Registra fallos y verifica ingenuo/log-lineal."""
import os, sys, json, time, numpy as np, warnings
from multiprocessing import Pool
sys.path.append(os.getcwd()); warnings.filterwarnings('ignore')
import bench as B, bench2 as B2
JOBS = [('fc_2014-12.npz', 36, 36), ('fc_2015-12.npz', 48, 36), ('fc_2016-12.npz', 60, 36), ('fc_2019-12.npz', 96, 60),
        ('fc_h48_2014-12.npz', 36, 48), ('fc_h48_2015-12.npz', 48, 48)]
if __name__ == '__main__':
    t0 = time.time(); log = {}
    for fn, n, h in JOBS:
        F = dict(np.load(fn)); ctxs = [B2.Y[i, :n] for i in range(B2.NS)]
        with Pool(10) as p: cl = dict(p.map(B2._cls, [(i, ctxs[i], h) for i in range(B2.NS)], chunksize=8))
        chk = {m: bool(np.array_equal(np.maximum(np.stack([cl[i][m][1] for i in range(B2.NS)]), 0).astype(np.float32), F[m])) for m in ['Ingenuo estacional', 'Log-lineal']}
        for m in ['SARIMA', 'ETS amortiguado']: F[m] = np.maximum(np.stack([cl[i][m][1] for i in range(B2.NS)]), 0).astype(np.float32)
        fal = {k: int(sum(k in cl[i] for i in range(B2.NS))) for k in ['_fallo_SARIMA', '_fallo_ETS']}
        np.savez_compressed(fn, **F); log[fn] = dict(verif_identicos=chk, fallos=fal, s=round(time.time() - t0))
        print(fn, log[fn], flush=True)
    json.dump(log, open('fix_clasicos_log.json', 'w'), indent=1); print('done')
