"""Cifras del análisis exploratorio usadas en el texto v11 (fuente única)."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import json
E = json.load(open('../paquete_original/exploracion_previa/resultados_ITS.json'))
T = json.load(open('../resultados/tendencia_ampliaciones.json')); L = json.load(open('../resultados/lugar_defuncion.json'))
V = json.load(open('../resultados/validacion_datos.json'))
r = lambda x, k=1: round(x, k)
def e(d): return [r(d['estimacion_pct'] if 'estimacion_pct' in d else d['pct'], 2), r(d['ic95_inf'] if 'ic95_inf' in d else d['ic_inf'], 2), r(d['ic95_sup'] if 'ic95_sup' in d else d['ic_sup'], 2)]
N = {'tendencia': {k: {p: e(E[k]['estimaciones'][p]) for p in E[k]['estimaciones']} for k in ['principal', 'HAC6', 'sin_2020', 'incluye_2024_corregido']}}
N['tendencia']['principal_diag'] = {'n': E['principal']['diagnostico']['n'], 'n_pre': E['principal']['diagnostico']['n_pre'], 'n_post': E['principal']['diagnostico']['n_post'],
    'LB12_p': r(T['diagnostico_principal']['ljung_box']['12']['lb_pvalue'], 3), 'LB24_p': r(T['diagnostico_principal']['ljung_box']['24']['lb_pvalue'], 3),
    'DW': r(T['diagnostico_principal']['durbin_watson'], 2), 'curvatura_p': r(E['principal']['diagnostico']['curvatura_pre_diagnostico_p'], 3)}
N['edad_conjunto'] = {g: {p: e(T['edad_conjunto'][g][p]) for p in T['edad_conjunto'][g]} for g in T['edad_conjunto']}
N['heterogeneidad_edad'] = {k: [r(v['chi2'], 2), float('%.3g' % v['p'])] for k, v in T['heterogeneidad_edad'].items()}
N['pares_holm'] = {k: float('%.3g' % v['p_holm']) for k, v in T['pares_cambio_pendiente'].items()}
for k in ['solo_residentes', 'registro_tardio_2023_media', 'registro_tardio_2023_maximo', 'cuasi_poisson_edad']:
    N[k] = {p: e(T[k][p]) for p in T[k]}
def per(key, p):
    d = L[key]['periodos'][p]; out = {}
    for a, b in [('ajustado_pct', 'aj_ic'), ('contrafactual_pct', 'cf_ic'), ('diferencia_pp', 'dif_ic')]:
        if a in d: out[a] = [r(d[a], 2), r(d[b][0], 2), r(d[b][1], 2)]
    return out
N['lugar'] = {}
for d in ['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE']:
    N['lugar'][d] = {'OR': {k: [r(v['OR'], 3), r(v['ic_inf'], 3), r(v['ic_sup'], 3)] for k, v in L[f'principal_{d}']['OR'].items()},
                     'periodos': {p: per(f'principal_{d}', p) for p in L[f'principal_{d}']['periodos']},
                     'anual_post': {p: per(f'anual_post_{d}', p) for p in L[f'anual_post_{d}']['periodos']},
                     'anios_2023_menos_2019': per(f'anios_{d}', '2023 menos 2019 (ajustado)'),
                     'anual_ajustado': {k: [r(x, 2) for x in v] for k, v in L[f'anios_{d}']['periodos']['anual'].items()},
                     'n_evento': L[f'principal_{d}']['n_evento'], 'n_total': L[f'principal_{d}']['n_defunciones']}
N['lugar_sens_2023'] = {k.replace('sens_', ''): per(k, '2023') for k in L if k.startswith('sens_') and 'periodos' in L[k] and '2023' in L[k]['periodos']}
N['lugar_sens_2020'] = {k.replace('sens_', ''): per(k, 'mar-dic 2020') for k in L if k.startswith('sens_') and 'periodos' in L[k] and 'mar-dic 2020' in L[k]['periodos']}
N['tasas_anuales'] = {k: {y: r(v, 2) for y, v in d.items()} for k, d in L['tasas_anuales_media_mensual'].items()}
N['tasas_ITS'] = {k[9:]: {p: e(v) for p, v in L[k].items()} for k in L if k.startswith('ITS_tasa_')}
N['multinomial'] = {c: {p: [r(v['RRR'], 3), r(v['ic_inf'], 3), r(v['ic_sup'], 3)] for p, v in d.items()} for c, d in L['sens_multinomial_base_hogar'].items()}
N['validacion'] = {'flujo': V['flujo'], 'duplicadas': V['filas_identicas_duplicadas'], 'no_residentes': V['ent_resid_fuera_1_32_en_analisis'],
                   'residual_2023': V['residual_2023'], 'completitud_por_sitio': V['completitud_mismo_anio_por_sitio_2017_2022'],
                   'completitud_por_edad': V['completitud_mismo_anio_por_edad_2017_2022'], 'reconstruccion': V['reconstruccion_max_dif_relativa']}

N['pares_holm_nivel'] = {k: float('%.3g' % v['p_holm']) for k,v in T['pares_cambio_nivel'].items()}
N['pares_cambio_nivel'] = T['pares_cambio_nivel']
N['covarianza_edades'] = T['covarianza_edades']
N['metodo_poisson'] = T['metodo_poisson']
N['registro_2023_escenario'] = T['registro_2023_escenario']
N['diagnosticos_ITS_tasas'] = L['diagnosticos_ITS_tasas']
N['tasas_metadata'] = L['tasas_metadata']
N['diagnostico_multinomial'] = L['diagnostico_multinomial']
N['metadatos_v11'] = L['metadatos_v11']
N['modelos_no_inferenciales'] = {k:v for k,v in L.items() if isinstance(v,dict) and v.get('inferencial') is False and 'n_celdas' in v}
N['diagnosticos_lugar'] = {k:v['diagnosticos'] for k,v in L.items() if isinstance(v,dict) and 'diagnosticos' in v}

json.dump(N, open('../resultados/numeros_v11.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: N['lugar'][k]['anios_2023_menos_2019'] for k in N['lugar']}, ensure_ascii=False))
print(json.dumps({k: N['lugar'][k]['periodos']['2023'] for k in N['lugar']}, ensure_ascii=False))
print(N['lugar']['Hogar']['anual_ajustado']['2019'], N['lugar']['Hogar']['anual_ajustado']['2020'], N['lugar']['Hogar']['anual_ajustado']['2023'])
print(N['edad_conjunto']['40-59']['pendiente_post'], N['edad_conjunto']['60+']['pendiente_post'], N['edad_conjunto']['20-39']['pendiente_post'], N['heterogeneidad_edad'], N['pares_holm'])
