"""Ablaciones de adaptación de Chronos-2 (corpus v2, origen dic-2016, h=36):
semillas (3), pasos (300 vs 1000), LoRA vs ajuste completo, y transferencia sin series de cáncer de mama."""
import os, sys, time, json, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
sys.path.append(os.getcwd()); from bench import metrics, Q
torch.set_num_threads(10)
H = 36; n = 60
C = pd.read_csv('corpus2_mortalidad_mensual.csv.gz', index_col=0)
Y = C[[str(i) for i in range(156)]].values.astype(np.float32)
ctxs = [Y[i, :n] for i in range(len(Y))]
is_bc = (C.grp == 'C. mama').values
base = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu')

def predict(pipe):
    out = []
    for b in range(0, len(ctxs), 256):
        q, _ = pipe.predict_quantiles([c[None] for c in ctxs[b:b+256]], prediction_length=H, quantile_levels=list(Q))
        out += [x[0].numpy() for x in q]
    return np.maximum(np.stack(out), 0)

RUNS = [('zero-shot', None), ('LoRA 1000 s0', dict(mode='lora', steps=1000, seed=0)),
        ('LoRA 1000 s1', dict(mode='lora', steps=1000, seed=1)), ('LoRA 1000 s2', dict(mode='lora', steps=1000, seed=2)),
        ('LoRA 300 s0', dict(mode='lora', steps=300, seed=0)),
        ('LoRA 1000 sin mama', dict(mode='lora', steps=1000, seed=0, excl_bc=True)),
        ('Completo 1000 s0', dict(mode='full', steps=1000, seed=0))]
rows, log = [], {}
t0 = time.time()
for name, cfg in RUNS:
    if cfg is None:
        pipe = base
    else:
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        tr = [c for c, b in zip(ctxs, is_bc) if not (cfg.get('excl_bc') and b)]
        t1 = time.time()
        pipe = base.fit(inputs=tr, prediction_length=12, finetune_mode=cfg['mode'],
                        learning_rate=1e-5 if cfg['mode'] == 'lora' else 1e-6, num_steps=cfg['steps'], batch_size=64,
                        min_past=24, context_length=n, output_dir=f'ft2_{name.replace(" ", "_")}',
                        remove_printer_callback=True, report_to='none', seed=cfg['seed'])
        log[name] = round(time.time() - t1)
    qs = predict(pipe)
    np.save(f'ft2_pred_{name.replace(" ", "_")}.npy', qs.astype(np.float32))
    for i, sid in enumerate(C.index):
        rows.append(dict(sid=sid, run=name, **metrics(ctxs[i], Y[i, n:n + H], qs[i][:, 1], qs[i])))
    pd.DataFrame(rows).to_csv('finetune2_resultados.csv.gz', index=False); json.dump(log, open('finetune2_log.json', 'w'))
    print(name, round(time.time() - t0), flush=True)
