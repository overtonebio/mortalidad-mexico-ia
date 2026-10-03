# Avisos de terceros, reconocimientos y citas

Este repositorio no incluye código, pesos ni datos de terceros salvo los datos indicados abajo, que viajan en los
paquetes de depósito y no en el repositorio. Las licencias de las bibliotecas y modelos se rigen por sus
propietarios; las versiones se fijan en `environments/`. Verifique cada licencia en su fuente antes de
redistribuir; esta lista no sustituye esos términos.

## Datos

| Fuente | Uso | Condiciones (verificar) |
|---|---|---|
| INEGI. Estadísticas de defunciones registradas (EDR), datos abiertos 2012-2024. https://www.inegi.org.mx/programas/edr/ | Corpus de 2 870 series y extracto C50 | Términos de libre uso del INEGI (https://www.inegi.org.mx/inegi/terminos.html): permiten copiar, distribuir y adaptar con crédito «Fuente: INEGI, …», aviso de las transformaciones y sin sugerir aval del INEGI. |
| INEGI. Catálogos de los datos abiertos de defunciones (sitio de ocurrencia, entidad, derechohabiencia, escolaridad, tamaño de localidad) | Validación de códigos | Ídem. |
| INEGI. Censo de Población y Vivienda 2020, tabulados del cuestionario básico | Proporciones regionales por edad | Ídem. |
| Naciones Unidas, DESA, División de Población. World Population Prospects 2024. https://population.un.org/wpp/ | Denominadores por edad | Condiciones de uso de WPP: **pendiente de confirmar** antes del depósito. |

Las transformaciones aplicadas por los autores (agregación, filtros, estandarización, modelos) no son del INEGI
ni de Naciones Unidas y no representan una postura de esas instituciones.

## Modelos preentrenados (no se distribuyen pesos)

- Chronos-2 (`amazon/chronos-2`). Ansari AF, Shchur O, Küken J, et al. Chronos-2: from univariate to universal
  forecasting. arXiv:2510.15821. 2025.
- Chronos-Bolt (`amazon/chronos-bolt-base`). Ansari AF, Stella L, Turkmen C, et al. Chronos: learning the
  language of time series. TMLR 2024. arXiv:2403.07815.
- TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`). Das A, Kong W, Sen R, Zhou Y. A decoder-only foundation model
  for time-series forecasting. ICML 2024. arXiv:2310.10688.

La revisión (commit) de los pesos usada en el análisis original es desconocida (véase `docs/MODELOS.md`).

## Métodos entrenados localmente y bibliotecas

- LightGBM: Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 2017.
- N-HiTS (biblioteca neuralforecast): Challu C, et al. NHITS: neural hierarchical interpolation for time series
  forecasting. AAAI 2023. doi:10.1609/aaai.v37i6.25854.
- LoRA (biblioteca peft, vía chronos-forecasting): Hu EJ, et al. LoRA: low-rank adaptation of large language
  models. arXiv:2106.09685. 2021.
- Modelos clásicos (SARIMA, ETS amortiguado): statsmodels.
- Cálculo y figuras: NumPy, pandas, SciPy, statsmodels, Matplotlib. Comparación de figuras: pypdfium2.
- Inferencia y entrenamiento (nivel 4): PyTorch, chronos-forecasting, timesfm, neuralforecast, lightgbm, peft,
  transformers.

## Herramientas de inteligencia artificial

Durante la preparación del código, la organización de este repositorio y la redacción se usaron herramientas de
IA generativa, declaradas en el manuscrito. No son autoras; los autores son responsables del contenido.
