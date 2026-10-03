"""LoRA con exclusión estricta (v5): mismo protocolo que finetune2.py ('LoRA 1000', semilla 0), pero el entrenamiento
excluye TODAS las series de cáncer de mama (C50) y TODAS las series de 'Todas las causas', que contienen las muertes C50.
Evalúa en todas las series. Salida: finetune_estricto_resultados.csv.gz, ft_estricto_pred.npy, log json."""
import os, sys, time, json, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
sys.path.append(os.getcwd()); from bench import metrics, Q
torch.set_num_threads(10); H = 36; n = 60
C = pd.read_csv('corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(np.float32); ctxs = [Y[i, :n] for i in range(len(Y))]
excl = C.grp.isin(['C. mama', 'Todas las causas']).values
base = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu')
torch.manual_seed(0); np.random.seed(0)
tr = [c for c, e in zip(ctxs, excl) if not e]; t0 = time.time()
pipe = base.fit(inputs=tr, prediction_length=12, finetune_mode='lora', learning_rate=1e-5, num_steps=1000, batch_size=64,
                min_past=24, context_length=n, output_dir='ft_estricto', remove_printer_callback=True, report_to='none', seed=0)
t_fit = time.time() - t0
out = []
for b in range(0, len(ctxs), 256):
    q, _ = pipe.predict_quantiles([c[None] for c in ctxs[b:b + 256]], prediction_length=H, quantile_levels=list(Q))
    out += [x[0].numpy() for x in q]
qs = np.maximum(np.stack(out), 0); np.save('ft_estricto_pred.npy', qs.astype(np.float32))
rows = [dict(sid=sid, run='LoRA 1000 estricto (sin C50 ni todas las causas)', **metrics(ctxs[i], Y[i, n:n + H], qs[i][:, 1], qs[i])) for i, sid in enumerate(C.index)]
pd.DataFrame(rows).to_csv('finetune_estricto_resultados.csv.gz', index=False)
json.dump(dict(n_train=len(tr), n_excluded=int(excl.sum()), fit_s=round(t_fit)), open('finetune_estricto_log.json', 'w'))
print('done', len(tr), int(excl.sum()), round(t_fit))
