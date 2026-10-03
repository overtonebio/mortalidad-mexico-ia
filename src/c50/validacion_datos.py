"""Paso 1 del plan del 30-sep-2026: validación de datos, reconstrucción de la serie y registro tardío."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd
from comun import *
d = leer_crudo(); R = {}
R['filas_crudas'] = len(d); R['filas_identicas_duplicadas'] = int(d.duplicated().sum())
w, pasos = filtrar(d, log=True); pasos.to_csv('../tablas/tablaS_flujo_registros.csv', index=False)
R['flujo'] = pasos.to_dict('records')
R['file_year_igual_anio_regis'] = bool((pd.to_numeric(d.file_year) == pd.to_numeric(d.anio_regis)).all())
R['causa_def'] = d.causa_def.value_counts().to_dict()
# residencia
R['ent_resid_fuera_1_32_en_analisis'] = w.loc[~w.residente, 'ent_resid'].value_counts().sort_index().astype(int).to_dict()
R['ent_resid_fuera_1_32_por_file_year'] = w[~w.residente].groupby(['file_year', 'ent_resid']).size().unstack(fill_value=0).to_dict('index')
# sitio de ocurrencia por archivo
st = w.groupby(['file_year', 'sitio_ocur']).size().unstack(fill_value=0); st.to_csv('../tablas/tablaS_sitio_codigo_por_archivo.csv')
R['codigos_sitio_observados'] = sorted(int(c) for c in st.columns); R['sin_grupo_sitio'] = int(w.sitio.isna().sum())
# reconstrucción de la serie
pm = poblacion_mensual(); fac = factor_2024(w)
cnt = w.groupby(['date', 'ag']).size().unstack(fill_value=0).reindex(IDX, fill_value=0)
s = pd.read_csv(SERIES, parse_dates=['fecha']).set_index('fecha')
rec = pd.DataFrame({'deaths': cnt.sum(1), 'deaths_adj': cnt.mul(fac, axis=0).sum(1), 'pop20': pm.sum(1),
                    'asmr': tee(cnt.mul(fac, axis=0), pm)})
for k, ags in {'20-39': [20, 25, 30, 35], '40-59': [40, 45, 50, 55], '60+': [60, 65, 70, 75, 80, 85]}.items():
    rec[f'asmr_{k}'] = tee(cnt.mul(fac, axis=0), pm, ags)
R['reconstruccion_max_dif_relativa'] = {c: float((rec[c] / s[c] - 1).abs().max()) for c in rec.columns}
assert max(R['reconstruccion_max_dif_relativa'].values()) < 1e-9
R['wpp_unidades'] = 'miles (x1000); población femenina >= 20 años, enero 2012 = %.0f' % pm.sum(1).iloc[0]
R['pesos_oms_suma'] = float(WHO.sum())
# registro tardío: rezago de registro por año de ocurrencia
w['rezago'] = w.anio_regis - w.anio_ocur
lag = w.groupby(['anio_ocur', 'rezago']).size().unstack(fill_value=0); lag.to_csv('../tablas/tablaS_rezago_registro.csv')
share = lag.div(lag.sum(1), axis=0)
ge2 = share.loc[:, [c for c in share.columns if c >= 2]].sum(1)
R['proporcion_registrada_mismo_anio'] = share[0].round(4).to_dict()
R['proporcion_registrada_ge2'] = ge2.round(5).to_dict()
hist = ge2.loc[2015:2021]
n23 = int(lag.loc[2023].sum())
R['residual_2023'] = {'ventana_historica': '2015-2021', 'proporcion_ge2_media': float(hist.mean()), 'proporcion_ge2_max': float(hist.max()),
                      'defunciones_2023_observadas': n23, 'faltantes_esperadas_media': float(n23 * hist.mean() / (1 - hist.mean())),
                      'faltantes_esperadas_max': float(n23 * hist.max() / (1 - hist.max())),
                      'efecto_TEE_2023_pct_media': float(100 * hist.mean() / (1 - hist.mean()))}
# completitud 2024 por lugar y por edad (registro en el mismo año, ocurrencias 2017-2022)
b = w[w.anio_ocur.between(2017, 2022)].assign(mismo=lambda x: x.anio_regis == x.anio_ocur)
R['completitud_mismo_anio_por_sitio_2017_2022'] = b.groupby('sitio').mismo.mean().round(4).to_dict()
b['edad3'] = pd.cut(b.age, [19, 39, 59, 200], labels=['20-39', '40-59', '60+'])
R['completitud_mismo_anio_por_edad_2017_2022'] = b.groupby('edad3', observed=True).mismo.mean().round(4).to_dict()
R['completitud_mismo_anio_global_2017_2022'] = float(b.mismo.mean())
json.dump(R, open('../resultados/validacion_datos.json', 'w'), indent=1, ensure_ascii=False, default=str)
w.to_pickle('../resultados/c50_analitico.pkl')
print(json.dumps({k: R[k] for k in ['filas_crudas', 'filas_identicas_duplicadas', 'flujo', 'file_year_igual_anio_regis',
      'ent_resid_fuera_1_32_en_analisis', 'codigos_sitio_observados', 'reconstruccion_max_dif_relativa', 'residual_2023',
      'completitud_mismo_anio_por_sitio_2017_2022', 'completitud_mismo_anio_por_edad_2017_2022', 'proporcion_registrada_ge2']}, ensure_ascii=False, default=str))
