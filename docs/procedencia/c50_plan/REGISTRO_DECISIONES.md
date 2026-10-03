# Registro de decisiones (análisis secundario exploratorio de C50, v10)

| Fecha y hora (CST) | Decisión | Motivo | Momento respecto de los resultados |
|---|---|---|---|
| 2026-09-30 13:15 | Plan fechado (PLAN_LUGAR_Y_AMPLIACIONES_2026-09-30.txt, SHA-256 25b7b417…af92) | Requisito de transparencia | Antes de todo ajuste nuevo; se conocían resultados originales, ITS previa y distribución anual cruda del lugar |
| 2026-09-30 13:17 | Descarga de catálogos oficiales de sitio de ocurrencia y entidad (13 archivos anuales) | Validar agrupación | Antes de ajustar |
| 2026-09-30 13:18 | Eliminar tabuladores de claves geográficas | INEGI incluye `\t` en ent_resid y en los catálogos | Antes de ajustar |
| 2026-09-30 13:18 | Conservar 19 filas idénticas | No hay identificador individual que pruebe duplicación | Antes de ajustar |
| 2026-09-30 13:18 | Asignar código 13 (2024) a unidad pública | Etiqueta oficial OPD IMSS BIENESTAR; sin registros C50 | Antes de ajustar |
| 2026-09-30 13:19 | Reproducción de la ITS previa en copia aislada | Auditoría | Diferencia máxima 3.7e-12 |
| 2026-09-30 13:19 | Adenda 1: cuasi-Poisson y 2023 inflado por registro tardío | Robustez de la escala y del registro residual | Antes de ajustar modelos nuevos |
| 2026-09-30 13:20 | Covarianza Driscoll-Kraay propia, verificada contra statsmodels (hac-groupsum) y contra HAC de serie única | Respetar dependencia temporal común a celdas y edades | Antes de ajustar modelos de lugar; ampliaciones de tendencia ajustadas a las 13:20 |
| 2026-09-30 13:21 | Adenda 2: proporciones anuales ajustadas (indicadores de año), promedios de periodo, ponderación 2024 por lugar | Precisar implementación | Antes de ajustar modelos de lugar |
| 2026-09-30 13:22-13:27 | Prueba técnica con 200 extracciones; ejecución final con 10 000 | Verificar código y tiempo | Los coeficientes no dependen del número de extracciones |
| 2026-09-30 | Figura 6 del manuscrito combina tendencia y lugar | Límite de GMM: 6 figuras | Después de resultados; no cambia estimaciones |
| 2026-09-30 | No usar derechohabiencia para lugar ni como estrato de riesgo | Cambios de codificación y sin denominadores | Fijado en la solicitud |

Horas tomadas de las marcas de tiempo de los archivos (validacion_datos.json 13:18:37; log_reproduccion.txt 13:19:10; ADENDA_1 13:19:36; tendencia_ampliaciones.json 13:20:35; ADENDA_2 13:21:19; lugar_defuncion.json 13:27:43).


## Correcciones de auditoría v11 — 30 de septiembre de 2026

Las filas anteriores se conservan como historia de v10. Las correcciones siguientes se decidieron después de conocer v10 y su auditoría, por autorización del usuario; no son un nuevo plan prospectivo. No se buscaron cortes ni variantes por significación.

| Corrección | Alcance y justificación |
|---|---|
| Lugar conocido | Excluir solo código99; conservar10 (vía pública) y12 (otro lugar). V10 excluía todo Otro/NE. Se recalcula esta sensibilidad con su composición propia. |
| Código99 separado | Se conserva el intento como no estimable: hay una entidad sin eventos, que impide una MLE logística finita con sus efectos fijos. Solo se publican flujo, descriptivos y diagnóstico; se suprimen OR, IC y p. |
| Guardias | Comprobar rango, niveles sin eventos/resto, convergencia, condición de la información, score y validez de covarianza antes de publicar inferencia. Los diagnósticos no cambian el modelo ni eliminan covariables para forzar estimación. |
| Modelo conjunto por edad | Conservar coeficientes y covarianza de Claude, factor temporal144/143 explícito. Informar como sensibilidad algebraica la corrección144/129 utilizada antes: cambia precisión, no coeficientes. La corrección elegida no es la opción cluster completa de statsmodels. |
| Contrastes por edad | Completar los tres pares de cambio de nivel con Holm, en familia separada de los tres pares de cambio de pendiente. No confundir heterogeneidad del cambio con diferencias de pendientes posteriores. |
| Poisson | Precisar que es Poisson con covarianza robusta de scores mensuales Bartlett12, offset población/12 y efectos temporales comunes por edad; conservar la clave heredada cuasi_poisson_edad por compatibilidad. No multiplicar otra vez por dispersión. |
| Residuo2023 | Presentar la inflación como escenario histórico de retraso, sujeto a seguimiento desigual y cola truncada; no como estimación validada de todo el faltante. |
| Tasas por lugar2024 | Descriptivas observadas sin factor global de retraso. La ITS de tasas por lugar termina en2023. La sensibilidad nacional2024 corregida y las probabilidades2024 ponderadas por lugar/mes conservan sus métodos y rótulos propios. |
| Dependencia y forma temporal | Añadir Ljung–Box6/12/24 y autocorrelación1 para tasas por lugar y modelos logísticos. Son diagnósticos aproximados; HAC no corrige la media mal especificada ni prueba equivalencia. |
| Multinomial | Conservar cuatro categorías y la covarianza agrupada por mes declarada por Claude; no llamarla HAC temporal. Guardar convergencia/condición/score. |
| Figuras | Mantener nombres internos para trazabilidad; root ubica la figuraC50 como figura5 y LoRA como6. Las tasas2024 por lugar se rotulan observadas. |

La estandarización principal de lugar se conserva sobre la distribución conjunta edad×entidad de los residentes1–32 de2012–2023, no sobre2017–2019. Las sensibilidades de lugar conocido, regiones y no residentes conservan sus poblaciones de referencia propias de v10.
