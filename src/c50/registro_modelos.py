"""Registro de todos los modelos ejecutados en el análisis secundario exploratorio (v11), con su función y archivo."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import json, pandas as pd
F = []
def a(grupo, id_, desc, per, corte, cov, rol, arch): F.append(dict(grupo=grupo, id=id_, especificacion=desc, periodo=per, corte=corte, covarianza=cov, funcion=rol, resultados=arch))
E = 'exploracion_previa/estimaciones_ITS.csv (reproducido: reproduccion/pk)'
its = 'log(TEE mensual) ~ tiempo + post + tiempo_post + 11 meses (MCO)'
a('Tendencia (reproducción)', 'T1', its + ', nacional', '2012-2023', 'mar 2020', 'HAC Bartlett 12, corrección n/(n-k)', 'principal exploratorio previo', E)
a('Tendencia (reproducción)', 'T2', its + ', nacional', '2012-2023', 'mar 2020', 'HAC 6', 'sensibilidad previa', E)
a('Tendencia (reproducción)', 'T3', its + ', nacional, excluye 2020 con distancias de calendario', '2012-2019 y 2021-2023', 'ene 2021', 'HAC 12 por distancia calendario', 'sensibilidad previa', E)
a('Tendencia (reproducción)', 'T4', its + ', nacional, 2024 corregido', '2012-2024', 'mar 2020', 'HAC 12', 'sensibilidad previa', E)
for g in ['20-39', '40-59', '60+']: a('Tendencia (reproducción)', f'T5_{g}', its + f', edad {g}, ajuste separado; Holm', '2012-2023', 'mar 2020', 'HAC 12', 'descriptivo previo', E)
a('Tendencia (reproducción)', 'T6', 'T1 + min(tiempo, corte)^2', '2012-2023', 'mar 2020', 'HAC 12', 'diagnóstico de curvatura previa', E)
T = 'tablas/tablaS_tendencia_ampliaciones.csv; resultados/tendencia_ampliaciones.json'
a('Tendencia (nuevo)', 'T7', 'Diagnóstico de residuos de T1: Ljung-Box 6/12/24, Durbin-Watson, ACF, Shapiro', '2012-2023', 'mar 2020', '-', 'diagnóstico', T)
a('Tendencia (nuevo)', 'T8', 'Modelo apilado de 3 edades, todos los términos por grupo; Wald 2 gl de igualdad (cambio de pendiente, nivel, pendiente previa); pares de cambio de nivel y pendiente con Holm, familias separadas', '2012-2023', 'mar 2020', 'Driscoll-Kraay 12 (mes), corrección T/(T-1)=144/143', 'plan 4.2', T)
a('Tendencia (nuevo)', 'T9', its + ', solo residentes 1-32', '2012-2023', 'mar 2020', 'HAC 12', 'plan 4.3', T)
a('Tendencia (nuevo)', 'T10', its + ', 2023 inflado por registro tardío residual (media 0.25 %)', '2012-2023', 'mar 2020', 'HAC 12', 'plan 4.5 / adenda 1', T)
a('Tendencia (nuevo)', 'T11', its + ', 2023 inflado por registro tardío residual (máximo 0.32 %)', '2012-2023', 'mar 2020', 'HAC 12', 'plan 4.5 / adenda 1', T)
a('Tendencia (nuevo)', 'T12', 'Poisson robusto: defunciones ~ tiempo + post + tiempo_post + mes + edad(14), offset log(mujeres-año)', '2012-2023', 'mar 2020', 'Driscoll-Kraay 12', 'adenda 1 (añadido)', T)
L = 'tablas/tablaS_lugar_*.csv; resultados/lugar_defuncion.json'
for d in ['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE']:
    base = f'Logística binomial por celdas mes x edad(14) x entidad(32): {d} vs resto; mes; '
    a('Lugar', f'L1_{d}', base + 'tiempo + post + tiempo_post', '2012-2023', 'mar 2020', 'Driscoll-Kraay 12', 'principal (hogar) / secundario', L)
    a('Lugar', f'L2_{d}', base + 'tiempo + indicadores mar-dic 2020, 2021, 2022, 2023', '2012-2023', 'mar 2020', 'Driscoll-Kraay 12', 'secundario prefijado', L)
    a('Lugar', f'L3_{d}', base + 'indicadores de año (proporción anual ajustada)', '2012-2023', '-', 'Driscoll-Kraay 12', 'descriptivo (adenda 2)', L)
for d in ['Hogar', 'Unidad pública', 'Unidad privada']:
    for s, desc, per, c, cv in [('abril2020', 'corte abril 2020', '2012-2023', 'abr 2020', 'DK 12'), ('HAC6', 'rezagos 6', '2012-2023', 'mar 2020', 'DK 6'),
                                ('lugar_conocido', 'excluye solamente código99; conserva10 y12', '2012-2023', 'mar 2020', 'DK 12'), ('region', 'región (4) en vez de entidad', '2012-2023', 'mar 2020', 'DK 12'),
                                ('no_residentes', 'incluye residencia fuera de 1-32 como categoría', '2012-2023', 'mar 2020', 'DK 12'),
                                ('2024_sin_ponderar', '2024 sin corrección', '2012-2024', 'mar 2020', 'DK 12'), ('2024_ponderado', '2024 ponderado 1/completitud por lugar y mes', '2012-2024', 'mar 2020', 'DK 12')]:
        a('Lugar (sensibilidad)', f'LS_{s}_{d}', f'L1 {d}: {desc}', per, c, cv, 'sensibilidad prefijada', L)
a('Lugar (intento no estimable)', 'LS_catalogo_99', 'No especificado99: separación estructural en entidad3; solo diagnóstico y descriptivos, sin OR/IC/p', '2012-2023', 'mar 2020', 'No procede inferencia', 'sensibilidad (g), intento conservado', L)
a('Lugar (sensibilidad)', 'LS_catalogo_10_12', 'L1 con agrupación alternativa: vía pública/otro10,12', '2012-2023', 'mar 2020', 'DK12, T/(T-1)', 'sensibilidad (g)', L)
a('Lugar (sensibilidad)', 'LS_multinomial', 'Multinomial individual (base hogar), mismas covariables que L1', '2012-2023', 'mar 2020', 'agrupado por mes', 'sensibilidad (e)', 'tablas/tablaS_lugar_multinomial.csv')
for r in ['Hogar', 'Unidad médica (pública + privada)', 'Otro/NE', 'Total']:
    a('Tasas por lugar', f'R_{r}', its + f', TEE de defunciones: {r}', '2012-2023', 'mar 2020', 'HAC12, n/(n-k)', 'complementario; diagnóstico residual informado', 'tablas/tablaS_tasas_por_lugar_ITS.csv')
a('Prueba técnica', 'X1', 'Ejecución completa de lugar_defuncion.py con 200 extracciones para verificar el código; sustituida por 10 000 extracciones (mismos coeficientes y errores estándar)', '-', '-', '-', 'prueba de código; no informada', '-')
a('Tasas descriptivas', 'R2024', 'TEE por lugar2024 observada, sin factor global; nacional2024 corregida se conserva solo en T4', '2024', '-', '-', 'descriptivo provisional', 'resultados/tasas_estandarizadas_por_lugar_mensual.csv')
P = pd.DataFrame(F); P.to_csv('../tablas/registro_modelos.csv', index=False); print(len(P), 'modelos registrados')
