"""P5 (v5, rutas v6). Compara las cifras publicadas (pronósticos guardados, v3/v4) con una nueva ejecución completa con semillas fijas.
Requiere: pronosticos/nueva_ejecucion_semillas/fc_<origen>.npz (bench2.py), pronosticos/nueva_ejecucion_semillas/fc_h48_<origen>.npz (bench2_h48.py),
entradas/p5_rerun_mama_detalle.csv (comparación guardada de la nueva ejecución de pipeline_tsfm.py en v5; no puede
recalcularse sin volver a correr los modelos). Ejecuta reproducir_benchmark.py sobre la nueva ejecución del benchmark.
Salida: documentos/tablas/tablaS14_reproducibilidad.csv y resultados/p5_rerun_detalle.csv"""
import json, os, subprocess, sys, numpy as np, pandas as pd
RB = 'pronosticos/nueva_ejecucion_semillas'; OUT = 'resultados/benchmark_nueva_ejecucion'
subprocess.run([sys.executable, 'codigo/auditoria/reproducir_benchmark.py', RB, OUT], check=True)
P = json.load(open('resultados/benchmark/numeros_repro.json')); Q = json.load(open(f'{OUT}/numeros_repro.json'))
MOD = ['LightGBM global', 'Ensamble (mediana)', 'TimesFM-2.5', 'Chronos-2', 'Chronos-Bolt', 'N-HiTS global', 'Log-lineal', 'ETS amortiguado', 'SARIMA', 'Ingenuo estacional']
rows = []
for m in MOD:
    rows.append(dict(Cifra='WQL relativo al log-lineal', Modelo=m, Publicada=P['rel'][m]['rel'], Nueva=Q['rel'][m]['rel']))
    rows.append(dict(Cifra='Cobertura 80 % (%)', Modelo=m, Publicada=P['cov'][m], Nueva=Q['cov'][m]))
    rows.append(dict(Cifra='Falsos positivos placebo (%)', Modelo=m, Publicada=P['plac_vol'][m]['FPR'], Nueva=Q['plac_vol'][m]['FPR']))
    rows.append(dict(Cifra='Semiancho mediano placebo (%)', Modelo=m, Publicada=P['plac_vol'][m]['semiancho_mediana'], Nueva=Q['plac_vol'][m]['semiancho_mediana']))
FP = json.load(open('resultados/benchmark/friedman_por_serie.json'))['por_serie_WQLmedio']['rangos']
FQ = json.load(open(f'{OUT}/friedman_por_serie.json'))['por_serie_WQLmedio']['rangos']
for m in MOD: rows.append(dict(Cifra='Rango Friedman por serie', Modelo=m, Publicada=FP[m], Nueva=FQ[m]))
# mama: comparación guardada (v5) de la nueva ejecución completa de la canalización de cáncer de mama
J = pd.read_csv('entradas/p5_rerun_mama_detalle.csv')
for _, r in J[(J.model == 'Ensamble (mediana)') & (J.year <= 2023)].iterrows():
    rows.append(dict(Cifra=f'Mama {r.target} {r.year}: diferencia (%)', Modelo='Ensamble (mediana)', Publicada=r.pscore_pub, Nueva=r.pscore_nuevo))
R = pd.DataFrame(rows); R['Diferencia'] = R.Nueva - R.Publicada
R.to_csv('resultados/p5_rerun_detalle.csv', index=False)
S = R.copy(); S['Publicada'] = S.Publicada.map(lambda x: f'{x:.3f}' if abs(x) < 2 else f'{x:.1f}'); S['Nueva'] = S.Nueva.map(lambda x: f'{x:.3f}' if abs(x) < 2 else f'{x:.1f}')
S['Publicada'] = S.Publicada.str.replace('-', '−'); S['Nueva'] = S.Nueva.str.replace('-', '−')
S['Diferencia'] = R.Diferencia.map(lambda x: f'{x:+.3f}' if abs(x) < 0.5 else f'{x:+.1f}').str.replace('-', '−')
S.to_csv('documentos/tablas/tablaS14_reproducibilidad.csv', index=False)
print(R.groupby('Cifra').Diferencia.agg(lambda d: np.abs(d).max()).round(3).to_string())
print(R[R.Modelo.isin(['SARIMA', 'ETS amortiguado', 'Ensamble (mediana)'])].round(3).to_string())
