# Datos: fuentes, obtención y verificación

## Distribución

Los datos se distribuyen como dos archivos adjuntos de la [entrega de datos `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014) y en el [depósito de datos de Zenodo](https://doi.org/10.5281/zenodo.23480195). Los contenidos descomprimidos se mantienen fuera del árbol Git en `data/external/`.

| Paquete | Contenido | Tamaño ZIP |
|---|---|---:|
| `GMM_v13_datos_y_entradas.zip` | `benchmark/datos`, `benchmark/entradas`, `c50/paquete_original`, `c50/fuentes_inegi` | 7.96 MB |
| `GMM_v13_pronosticos_NPZ.zip` | `benchmark/pronosticos/publicados` (6) y `benchmark/pronosticos/nueva_ejecucion_semillas` (6), con su manifiesto original | 135.29 MB |

La [guía de descarga y ejecución](DESCARGA_Y_EJECUCION.md) contiene los enlaces, la comprobación SHA-256, los comandos de extracción y los pasos para ejecutar los análisis. Los archivos originales conservan sus hashes y se acompañan del paquete de procedencia WPP; véase `PUBLICACION.md`.

Después de extraer ambos paquetes en la misma carpeta, el comando

```sh
python scripts/gmm.py verificar --data-dir data/external
```

comprueba 109 archivos contra `configs/datos_esperados.json` (SHA-256 y tamaño), las 108 matrices NPZ
(nueve modelos × 12 archivos; forma 2 870 × H × 3; valores finitos) y el corpus (2 870 × 156, finito y no
negativo). Una copia íntegra del ZIP original del archivo científico v13 puede verificarse con
`python scripts/gmm.py verificar-archivo --archivo <Analisis_y_pronosticos>` (manifiesto original de 292 archivos).

## Catálogo

`CATALOGO_DATOS.csv` describe, por archivo o grupo de archivos: fuente, URL oficial, periodo, unidad de
observación, variables, transformaciones y condiciones de reutilización verificables.

## Condiciones de reutilización

- **INEGI** (EDR, catálogos y Censo 2020): los términos de libre uso del INEGI permiten copiar, distribuir y
  adaptar la información, con crédito al INEGI y aviso de las transformaciones, sin sugerir aval institucional
  (https://www.inegi.org.mx/inegi/terminos.html, consultado el 1 de octubre de 2026). Crédito sugerido:
  «Fuente: INEGI, Estadísticas de defunciones registradas (datos abiertos), 2012-2024. Procesamiento de los
  autores».
- **WPP 2024** (Naciones Unidas): © 2024 United Nations,
  [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/), confirmada en el catálogo oficial y en el
  archivo Excel correspondiente. El extracto de México se reconstruyó byte por byte desde el CSV oficial;
  357/357 filas, sin diferencias. [Fuente, filtros, unidades, hashes y procedimiento](WPP.md).
- **Documentación y productos propios de los autores** (corpus, entradas, pronósticos, resultados):
  CC BY 4.0, conservando los derechos y créditos de las fuentes. Código propio: MIT. Véase [LICENCIAS.md](../LICENCIAS.md).

La accesibilidad pública de un archivo no se interpretó como permiso de redistribución.

## Privacidad

El extracto C50 conserva variables públicas de los datos abiertos (claves geográficas de residencia, edad,
fechas, escolaridad, estado civil, asistencia médica, sitio de ocurrencia, derechohabiencia). No contiene nombres,
CURP ni domicilios. No se eliminan filas idénticas, porque no hay identificador individual que pruebe
duplicación. La base analítica en pickle (`c50_analitico.pkl`) no se distribuye: `validacion_datos.py` la
reconstruye.

## Datos que no se incluyen

- Microdatos nacionales completos de EDR 2012-2024 (necesarios para reconstruir el corpus desde cero; se
  descargan del INEGI con las URL de `c50/fuentes_inegi/urls_datos_abiertos_inegi.json`). El código de
  construcción está en `src/benchmark/construccion_datos/` (nivel 4).
- Pesos de los modelos preentrenados (véase `MODELOS.md`).

La carpeta histórica `pronosticos/publicados` identifica los seis pronósticos usados como referencia en el manuscrito; no afirma publicación previa. Los otros seis corresponden a una ejecución con semillas.
