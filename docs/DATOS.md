# Datos: fuentes, obtención y verificación

## Distribución

Los datos se distribuyen como dos archivos adjuntos de la [entrega privada `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014), accesible con la misma cuenta autorizada para el repositorio. Los contenidos descomprimidos se mantienen fuera del árbol Git en `data/external/`.

| Paquete | Contenido | Tamaño ZIP |
|---|---|---:|
| `GMM_v13_datos_y_entradas.zip` | `benchmark/datos`, `benchmark/entradas`, `c50/paquete_original`, `c50/fuentes_inegi` | 7.96 MB |
| `GMM_v13_pronosticos_NPZ.zip` | `benchmark/pronosticos/publicados` (6) y `benchmark/pronosticos/nueva_ejecucion_semillas` (6), con su manifiesto original | 135.29 MB |

La [guía de descarga y ejecución](DESCARGA_Y_EJECUCION.md) contiene los enlaces, la comprobación SHA-256, los comandos de extracción y los pasos para ejecutar los análisis. La distribución privada no asigna DOI ni sustituye el depósito futuro descrito en `PUBLICACION.md`.

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
- **WPP 2024** (Naciones Unidas): la [API oficial de la División de Población](https://population.un.org/dataportalapi/index.html)
  declara CC BY 3.0 IGO en su [especificación OpenAPI](https://population.un.org/dataportalapi/swagger/DataPortalOpenAPISpecificationv1.0/swagger.json),
  consultada el 1 de octubre de 2026 (extracto de metadatos en `docs/procedencia/licencia_portal_ONU.json`).
  Falta registrar el archivo de población original y la transformación que produjo `wpp_mex_female_age5.csv`;
  no se atribuye automáticamente esa licencia al extracto sin vincular su procedencia. Mantenga la atribución
  de Naciones Unidas y sus condiciones, separadas de la licencia que se elija para productos propios.
  Obtención: https://population.un.org/wpp/ (población por sexo y grupo quinquenal, variante media, México, mujeres).
- **Productos derivados de los autores** (corpus, entradas, pronósticos, resultados): sin licencia hasta la
  decisión de los autores (`LICENSING_PENDING.md`).

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
