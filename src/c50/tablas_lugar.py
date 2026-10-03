"""Tablas completas de lugar de defunción a partir de ../resultados/lugar_defuncion.json."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import json, pandas as pd, numpy as np
R = json.load(open('../resultados/lugar_defuncion.json'))
f1 = []; f2 = []
for k, r in R.items():
    if not isinstance(r, dict) or 'OR' not in r: continue
    mod, _, des = k.rpartition('_')
    for p, e in r['OR'].items():
        f1.append({'analisis': mod, 'desenlace': des, 'parametro': p, 'OR': e['OR'], 'ic95_inf': e['ic_inf'], 'ic95_sup': e['ic_sup'], 'p': e['p'],
                   'n_defunciones': r['n_defunciones'], 'n_evento': r['n_evento'], 'celdas': r['n_celdas'], 'convergencia': r['convergencia'],
                   'wald_nivel_pendiente_p': r.get('wald_nivel_pendiente', {}).get('p')})
    for per, e in r.get('periodos', {}).items():
        if per == 'anual': continue
        f2.append({'analisis': mod, 'desenlace': des, 'periodo': per, 'ajustado_pct': e.get('ajustado_pct'),
                   'aj_ic_inf': (e.get('aj_ic') or [None, None])[0], 'aj_ic_sup': (e.get('aj_ic') or [None, None])[1],
                   'contrafactual_pct': e.get('contrafactual_pct'), 'cf_ic_inf': (e.get('cf_ic') or [None, None])[0], 'cf_ic_sup': (e.get('cf_ic') or [None, None])[1],
                   'diferencia_pp': e.get('diferencia_pp'), 'dif_ic_inf': e['dif_ic'][0] if 'dif_ic' in e else None, 'dif_ic_sup': e['dif_ic'][1] if 'dif_ic' in e else None})
pd.DataFrame(f1).to_csv('../tablas/tablaS_lugar_razones_momios.csv', index=False)
pd.DataFrame(f2).to_csv('../tablas/tablaS_lugar_proporciones_estandarizadas.csv', index=False)
an = []
for des in ['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE']:
    for y, v in R[f'anios_{des}']['periodos']['anual'].items(): an.append({'desenlace': des, 'anio': int(y), 'ajustado_pct': v[0], 'ic95_inf': v[1], 'ic95_sup': v[2]})
pd.DataFrame(an).to_csv('../tablas/tablaS_lugar_proporcion_ajustada_anual.csv', index=False)
m = []
for des, d in R['sens_multinomial_base_hogar'].items():
    for p, e in d.items(): m.append({'categoria_vs_hogar': des, 'parametro': p, **e})
pd.DataFrame(m).to_csv('../tablas/tablaS_lugar_multinomial.csv', index=False)
t = []
for k, r in R.items():
    if k.startswith('ITS_tasa_'):
        for p, e in r.items(): t.append({'tasa': k[9:], 'parametro': p, 'estimacion_pct': e['pct'], 'ic95_inf': e['ic_inf'], 'ic95_sup': e['ic_sup'], 'p': e['p']})
pd.DataFrame(t).to_csv('../tablas/tablaS_tasas_por_lugar_ITS.csv', index=False)
pd.DataFrame(R['tasas_anuales_media_mensual']).to_csv('../tablas/tablaS_tasas_por_lugar_anuales.csv', index_label='anio')

# Invalid fits are retained only in diagnostics and descriptive tables, never OR/CI/p.
diag=[]; desc=[]
for key,v in R.items():
    if not isinstance(v,dict) or 'diagnosticos' not in v: continue
    d=v['diagnosticos']; diag.append({'modelo':key,'inferencial':v['inferencial'],
        'n_defunciones':v['n_defunciones'],'n_evento':v['n_evento'],'n_celdas':v['n_celdas'],
        'n_parametros':d['n_parametros'],'rango_diseno':d['rango_diseno'],
        'convergencia':v['convergencia'],'H_condicion':d.get('H_condicion'),
        'score_max_abs':d.get('score_max_abs'),'estado':d['estado'],
        'separacion_estructural':json.dumps(d['separacion_estructural'],ensure_ascii=False),
        'LB12_p':d.get('residuos_mensuales',{}).get('ljung_box',{}).get('12',{}).get('lb_pvalue')})
    for anio,entry in v.get('descriptivos_anuales',{}).items():
        desc.append({'modelo':key,'anio':int(anio),**entry})
pd.DataFrame(diag).to_csv('../tablas/tablaS_lugar_diagnosticos.csv',index=False)
pd.DataFrame(desc).to_csv('../tablas/tablaS_lugar_no_estimable_descriptivos.csv',index=False)
td=[]
for lugar,d in R['diagnosticos_ITS_tasas'].items():
    for lag,v in d['ljung_box'].items(): td.append({'lugar':lugar,'rezago':int(lag),**v,
        'n_meses':d['n_meses'],'k':d['k'],'autocorrelacion_1':d['autocorrelacion_1']})
pd.DataFrame(td).to_csv('../tablas/tablaS_tasas_por_lugar_diagnosticos.csv',index=False)
