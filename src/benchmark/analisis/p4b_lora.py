"""P4b (v5). Métricas LoRA armonizadas (origen 2016-12, h = 36, 2 870 series).
Definiciones únicas: WQL relativo = media geométrica por serie de WQL_variante / WQL_zero-shot;
'% de series que mejoran' = proporción con WQL_variante < WQL_zero-shot; Wilcoxon de rangos con signo (bilateral) sobre
log(WQL_variante / WQL_zero-shot) por serie. Subconjuntos: series C. mama; series 'Todas las causas'; resto.
Entradas: entradas/finetune2_resultados.csv.gz (publicado) y entradas/finetune_estricto_resultados.csv.gz (nuevo)."""
import numpy as np, pandas as pd, json, os
from scipy import stats
F = pd.read_csv('entradas/finetune2_resultados.csv.gz')
p = 'entradas/finetune_estricto_resultados.csv.gz'
if os.path.exists(p): F = pd.concat([F, pd.read_csv(p)[['sid', 'run', 'MASE', 'WQL', 'cov80']]])
C = pd.read_csv('datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)[['grp', 'level']]
W = F.pivot_table(index='sid', columns='run', values='WQL').join(C)
CV = F.pivot_table(index='sid', columns='run', values='cov80')
rows = []
for r in [c for c in F.run.unique() if c != 'zero-shot']:
    lr = np.log(W[r] / W['zero-shot'])
    def summ(mask):
        x = lr[mask]; return dict(n=int(mask.sum()), WQL_rel_geo=float(np.exp(x.mean())), mejora_media_pct=float((1 - np.exp(x.mean())) * 100),
                                  pct_series_mejoran=float((x < 0).mean() * 100), wilcoxon_p=float(stats.wilcoxon(x).pvalue) if mask.sum() > 10 else np.nan)
    a = summ(np.ones(len(W), bool)); m = summ((W.grp == 'C. mama').values); t = summ((W.grp == 'Todas las causas').values)
    o = summ(~W.grp.isin(['C. mama', 'Todas las causas']).values)
    rows.append(dict(variante=r, **{f'todas_{k}': v for k, v in a.items()}, **{f'mama_{k}': v for k, v in m.items()},
                     **{f'todascausas_{k}': v for k, v in t.items()}, **{f'resto_{k}': v for k, v in o.items()},
                     WQL_rel_media_aritm_de_razones=float((W[r] / W['zero-shot']).mean()), cob80=float(CV[r].mean())))
R = pd.DataFrame(rows); R.to_csv('resultados/p4b_lora_armonizado.csv', index=False)
pd.set_option('display.width', 250)
print(R[['variante', 'todas_n', 'todas_WQL_rel_geo', 'todas_pct_series_mejoran', 'todas_wilcoxon_p', 'mama_n', 'mama_WQL_rel_geo', 'mama_pct_series_mejoran', 'mama_wilcoxon_p', 'todascausas_WQL_rel_geo', 'WQL_rel_media_aritm_de_razones', 'cob80']].round(4).to_string())
print('zero-shot cob80', CV['zero-shot'].mean())
