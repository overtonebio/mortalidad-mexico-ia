# mortalidad-mexico-ia

Código del estudio **«SI: Inteligencia artificial para estimar la mortalidad esperada en México: evaluación
comparativa y aplicación al cáncer de mama»** (manuscrito v20, 17 de octubre de 2026; preparado para envío a *Gaceta Médica de
México*, número especial de inteligencia artificial; **no publicado** a la fecha de esta preparación).

> Estado: versión de software **propuesta y no publicada** (borrador para revisión). Sin licencia definitiva:
> véase [LICENSING_PENDING.md](LICENSING_PENDING.md). No existe todavía DOI ni repositorio público.

## Versión vigente del manuscrito

La actualización **0.2.0-rc.1**, preparada el 19 de octubre de 2026, corresponde al **manuscrito v20**.
Incluye la ampliación ajustada por localidad, las tablas suplementarias renumeradas y los rótulos finales de
las doce figuras. El análisis base v13, sus entradas y sus pronósticos se conservan íntegros.

Desde una copia nueva del repositorio, descargue los mismos dos paquetes de datos indicados abajo y prepare
el entorno C50. Después ejecute:

```sh
.venv-c50/bin/python scripts/gmm_v20.py verificar --data-dir data/external
.venv-c50/bin/python scripts/gmm_v20.py figuras --data-dir data/external --output-dir ejecuciones/figuras_v20
.venv-c50/bin/python scripts/gmm_v20.py localidad --data-dir data/external --output-dir ejecuciones/localidad_v20
```

Use directorios de salida nuevos y separados del código y los datos. No hace falta descargar una adenda
adicional: todos los insumos necesarios ya están en los dos ZIP v13; el código y las referencias nuevos están
en `src/v20/` y `results/reference/v20/`. La adenda original se conserva aparte como antecedente de depósito.
La verificación del 19 de octubre reprodujo 12/12 figuras con texto y render idénticos a los PDF del envío,
y 20/20 CSV de localidad idénticos byte a byte. Véase [Reproducción v20](docs/REPRODUCCION_V20.md).

El repositorio sigue **privado**. La publicación abierta, las licencias y el DOI siguen pendientes; un enlace
privado no da acceso a revisores que no tengan permiso. [Estado de publicación](docs/PUBLICACION.md).

## Qué contiene

| Carpeta | Contenido |
|---|---|
| `src/benchmark/` | Evaluación comparativa de 10 métodos en 2 870 series mensuales de mortalidad (métricas, intervalos, detección, cosecha, Friedman, LoRA, derechohabiencia) y scripts de generación de pronósticos y entrenamiento (nivel 4) |
| `src/c50/` | Análisis exploratorios de cáncer de mama (C50): tendencia segmentada, lugar de defunción, APVP y tamaño de localidad |
| `src/figuras/` | Generadores históricos de la base v13 |
| `src/v20/`, `scripts/gmm_v20.py` | Localidad ajustada y las doce figuras del manuscrito vigente |
| `scripts/gmm.py` | Ejecutor por niveles con `--help`, verificación de entradas y directorios de salida separados |
| `configs/` | Pasos, correspondencia de figuras, hashes de entradas esperadas y manifiestos originales |
| `environments/` | Requisitos por entorno (benchmark y C50 por separado) y versiones históricas declaradas |
| `results/reference/` | Resultados de referencia del archivo científico v13 (**inmutables**) |
| `docs/` | Documentación técnica, de datos, modelos, publicación, trazabilidad y procedencia |
| `tests/` | Pruebas ligeras con datos sintéticos; sin descargas ni GPU |

Los datos procesados de entrada y los 12 archivos de pronósticos NPZ se descargan de la
[entrega privada `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014), en **Assets**.
La [guía de descarga y ejecución](docs/DESCARGA_Y_EJECUCION.md) explica cómo extraer ambos ZIP en
`data/external/`, verificar sus hashes y ejecutar el benchmark, C50 y las figuras.

## Resumen para investigadores médicos

El estudio compara modelos clásicos y de inteligencia artificial para estimar cuántas defunciones «se esperaban»
en México si no hubiera ocurrido la pandemia, y aplica esa estimación al cáncer de mama en mujeres de 20 años o
más. Este repositorio permite comprobar que las tablas y figuras del artículo se obtienen de los pronósticos y
datos guardados, y repetir los análisis exploratorios de cáncer de mama a partir del extracto público de
defunciones del INEGI. No vuelve a entrenar los modelos de IA (eso requiere pesos de modelos, otro entorno y
más tiempo), aunque conserva y documenta su código. Explicación ampliada:
[docs/PARA_INVESTIGADORES_MEDICOS.md](docs/PARA_INVESTIGADORES_MEDICOS.md).

## Instalación desde una carpeta limpia

Requisitos: Python 3.11 (benchmark) y Python 3.12 (C50 y figuras) con `pip`. Espacio estimado (no medido): ≈ 1.5 GB para dos entornos, los datos y una ejecución de cada nivel.

```sh
# Entorno del benchmark (nivel 3)
python3.11 -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r environments/benchmark-recalculo.txt

# Entorno C50 y figuras (niveles 2 y 3)
python3.12 -m venv .venv-c50
.venv-c50/bin/python -m pip install -r environments/c50-figuras.txt
```

Descargue los dos ZIP de la [entrega privada](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014) y siga
[docs/DESCARGA_Y_EJECUCION.md](docs/DESCARGA_Y_EJECUCION.md) para extraerlos en `data/external/`.

## Comandos de la base histórica v13

| Nivel | Qué hace | Comando | Tiempo medido* |
|---|---|---|---|
| 1 | Integridad SHA-256, formatos, dimensiones y finitud | `python scripts/gmm.py verificar --data-dir data/external` | ≈ 1 s |
| 2 | Regenera las 12 figuras desde resultados guardados | `.venv-c50/bin/python scripts/gmm.py figuras --data-dir data/external --output-dir ejecuciones/figuras` | ≈ 7 s |
| 3 | Recalcula el benchmark desde los NPZ | `.venv-benchmark/bin/python scripts/gmm.py benchmark --data-dir data/external --output-dir ejecuciones/benchmark` | ≈ 50 s (48 y 53 s en dos ejecuciones) |
| 3 | Recalcula los análisis C50 (10 000 simulaciones, 2 hilos) | `.venv-c50/bin/python scripts/gmm.py c50 --data-dir data/external --output-dir ejecuciones/c50 --figuras` | ≈ 4 min (236-241 s; 227-231 s en `lugar_defuncion`) |
| — | Compara una ejecución con `results/reference` | `python scripts/gmm.py comparar --output-dir ejecuciones/benchmark` | segundos |
| 4 | Inferencia y entrenamiento: solo documentación | `python scripts/gmm.py entrenamiento` | no ejecutado |

\* Medido el 1 de octubre de 2026 en macOS arm64 con 11 núcleos lógicos y 18 GB de RAM, sin GPU.

Las figuras pueden regenerarse también desde salidas de nivel 3:
`... figuras --resultados-benchmark ejecuciones/benchmark --resultados-c50 ejecuciones/c50`.
Para comparar por render con los PDF del envío v13 instale `environments/comparacion-figuras.txt` y añada
`--figuras-v13 <carpeta con Figura_*.pdf>` a `comparar`.

**Qué significa cada nivel.** El nivel 1 comprueba integridad, no validez científica. El nivel 2 regenera
figuras a partir de resultados ya calculados y no es una reproducción completa. El nivel 3 recalcula métricas,
intervalos, detección, tablas y los análisis C50 desde las entradas disponibles; no regenera los pronósticos.
El nivel 4 (generar pronósticos con Chronos-2, Chronos-Bolt, TimesFM-2.5, LightGBM, N-HiTS y los modelos
clásicos, y adaptar Chronos-2 con LoRA) **no se ejecutó** en esta preparación; está documentado en
[docs/ENTRENAMIENTO.md](docs/ENTRENAMIENTO.md). Ningún comando ordinario descarga modelos ni entrena.

Lo efectivamente comprobado y sus límites: [docs/REPRODUCCION.md](docs/REPRODUCCION.md) y
[docs/MATRIZ_TRAZABILIDAD.csv](docs/MATRIZ_TRAZABILIDAD.csv) (figuras 1-6, S1-S6, tablas 1-2 y S1-S25).

## Entradas y salidas

- Entradas: `--data-dir` con `benchmark/datos`, `benchmark/entradas`, `benchmark/pronosticos` y
  `c50/paquete_original`, `c50/fuentes_inegi`. Cada archivo se verifica contra `configs/datos_esperados.json`.
- Salidas: siempre en un `--output-dir` nuevo o vacío; se rechaza escribir en `results/reference` o en el
  repositorio. Cada ejecución guarda `registro_ejecucion.json` (versiones, plataforma, comandos, tiempos) y un
  registro por paso en `logs/`.

## Limitaciones

- Los pronósticos NPZ, los resultados del ajuste fino y varios CSV de `benchmark/entradas` proceden de etapas
  cuyo código está presente solo en parte o cuyo generador no se localizó (tablas S3, S5-S8 y piloto S3). Sus cifras
  se cotejaron con el manuscrito, pero no se regeneraron. La revisión de Codex añadió una reconstrucción comprobada de S2A
  y de los conteos de S1 (véase docs/REPRODUCCION.md).
- La revisión exacta de los pesos de Chronos-2, Chronos-Bolt y TimesFM-2.5 usada en el análisis original es
  desconocida.
- La figura 5 y las S4-S6 usan Arial si está instalada (macOS); en otros sistemas cambia la tipografía.
- Los análisis C50 son exploratorios y ecológicos; no sustentan inferencias causales.

## Cómo citar

Mientras no exista una versión publicada, cite el artículo cuando se publique y este código como software no
publicado (metadatos en [CITATION.cff](CITATION.cff)):

> Zavala-Reyes D, Portales-Pérez DP. mortalidad-mexico-ia: código del estudio sobre mortalidad esperada en
> México y cáncer de mama (versión propuesta, no publicada). 2026.

## Autores

- Daniel Zavala-Reyes — ORCID [0000-0002-4730-5149](https://orcid.org/0000-0002-4730-5149). Centro de
  Investigación en Ciencias de la Salud y Biomedicina, Universidad Autónoma de San Luis Potosí. Diseño, análisis
  y redacción.
- Diana P. Portales-Pérez — ORCID [0000-0001-8063-7869](https://orcid.org/0000-0001-8063-7869). Laboratorio de
  Inmunología y Biología Celular y Molecular, Facultad de Ciencias Químicas, Universidad Autónoma de San Luis
  Potosí. Revisión y correspondencia.

Sin financiamiento ni conflictos de intereses, según declaración de los autores. Herramientas de IA usadas en la
preparación del código y del texto se declaran en el manuscrito; no son autoras. Reconocimientos y licencias de
terceros: [AVISOS_TERCEROS.md](AVISOS_TERCEROS.md).


## English summary

Code for the study *"SI: Artificial intelligence for estimating expected mortality in Mexico: comparative
evaluation and application to breast cancer"* (manuscript v20, prepared for submission, not yet published). The repository
compares classical, global machine-learning and pretrained foundation forecasting models on 2 870 monthly
Mexican mortality series (INEGI, 2012-2024) and applies the forecasts to breast cancer (ICD-10 C50) in women
aged ≥ 20 years, together with exploratory trend, place-of-death, premature-mortality (YPLL) and
locality-size analyses. `scripts/gmm.py` runs four levels: (1) integrity and format checks; (2) regeneration
of the 12 manuscript figures from stored results; (3) recalculation of benchmark metrics from stored forecasts
and of the C50 analyses from the public microdata extract; (4) model inference and training, documented but
not executed in this preparation. Every run writes to a new `--output-dir`; reference results are never
modified. Data and forecast packages are available as assets of the private `datos-v13-20261014` release.
See [download and execution instructions](docs/DESCARGA_Y_EJECUCION.md). A GitHub account with access to
this repository is required. Public software publication, licensing and DOI remain pending.
