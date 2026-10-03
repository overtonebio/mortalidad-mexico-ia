"""Tabla 1 y tabla S15 desde resultados guardados del benchmark (sin generar documentos Word).

Extracción literal de las líneas 6-25 de generar_manuscrito.py (paquete_GMM_SI_IA_v10_revision/codigo/documentos/,
SHA-256 0a1e331a9b4a76284999a7f3df38f05568cc1f7e5a06947dfeac1c8d4ddd3ae4). Cambios: se omiten la importación del
constructor de Word y la construcción del documento; se añade la escritura de tabla1_manuscrito.csv (no existía
como CSV en el archivo científico v13). Ejecutar desde la raíz de trabajo del benchmark (scripts/gmm.py lo hace).
"""
import os, sys, json, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(D, '..', 'auditoria'))
import limpiar_tablas as LT
X = json.load(open('resultados/numeros_v9.json'))
t1 = pd.read_csv('resultados/benchmark/tabla1_repro.csv').sort_values('Rango WQL (por serie)')
m = lambda x: x.replace('-', '−')
rows = []
for _, r in t1.iterrows():
    red = 'referencia' if r.Modelo == 'Log-lineal' else m(f"{(1 - r.WQL_rel) * 100:.1f} ({(1 - r.IC_hi) * 100:.1f} a {(1 - r.IC_lo) * 100:.1f})")
    d1 = X['p2']['D1'][r.Modelo]
    rows.append({'Modelo': r.Modelo, 'Familia': r.Familia, 'Rango WQL': f"{r['Rango WQL (por serie)']:.2f}",
                 'Reducción de WQL vs log-lineal, % (IC 95 %)': red, 'Cobertura observada del IP 80 % (%)': f"{r.Cobertura80:.1f}",
                 'Falsas alarmas, IP bilateral 90 % (%)': f"{r.FPR:.1f}", 'Semiancho IP 90 % (%)': f"{r.Semiancho:.0f}",
                 'Falsas alarmas superiores, unilateral (%)': f"{d1['fa_sup']:.1f}", 'Detección de +20 %, unilateral (%)': f"{d1['det20']:.1f}"})
tab1c = pd.DataFrame(rows); assert LT.verificar(tab1c) == [], LT.verificar(tab1c)
NMOD = {'LightGBM global': 'LightGBM', 'Ensamble (mediana)': 'Ensamble', 'N-HiTS global': 'N-HiTS', 'ETS amortiguado': 'ETS', 'Ingenuo estacional': 'Ingenuo estacional'}
tab1c['Modelo'] = tab1c['Modelo'].map(lambda v: NMOD.get(v, v)); tab1c['Familia'] = tab1c['Familia'].replace({'Ensamble': 'Ensamble (clásicos + fundacionales)'}); tab1c.to_csv('documentos/tablas/tablaS15_desempeno_completo.csv', index=False)
# Tabla 1 simplificada (v8): la versión completa pasa a la tabla S15
tab1 = tab1c[['Modelo', 'Familia', 'Rango WQL', 'Reducción de WQL vs log-lineal, % (IC 95 %)', 'Cobertura observada del IP 80 % (%)',
              'Falsas alarmas superiores, unilateral (%)', 'Detección de +20 %, unilateral (%)']].rename(columns={
    'Rango WQL': 'Rango (1 = mejor)', 'Reducción de WQL vs log-lineal, % (IC 95 %)': 'Reducción del error frente al log-lineal, % (IC 95 %)',
    'Cobertura observada del IP 80 % (%)': 'Cobertura del intervalo de 80 % (%)', 'Falsas alarmas superiores, unilateral (%)': 'Falsas alarmas, 2017-2019 (%)',
    'Detección de +20 %, unilateral (%)': 'Exceso de 20 % detectado (%)'})
tab1['Familia'] = tab1['Familia'].replace({'Ensamble (clásicos + fundacionales)': 'Clásicos e IA'})
tab1.to_csv('documentos/tablas/tabla1_manuscrito.csv', index=False)
