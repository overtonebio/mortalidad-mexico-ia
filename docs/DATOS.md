# Datos: fuentes, obtención y verificación

## Distribución

Los datos no se versionan en git. Se preparan dos paquetes para depósito separado (véase `PUBLICACION.md`):

| Paquete | Contenido | Tamaño aprox. |
|---|---|---|
| Datos y entradas | `benchmark/datos`, `benchmark/entradas`, `c50/paquete_original`, `c50/fuentes_inegi` | 21 MB sin comprimir |
| Pronósticos NPZ | `benchmark/pronosticos/publicados` (6) y `benchmark/pronosticos/nueva_ejecucion_semillas` (6), con su manifiesto original | 129 MB |

Antes de que existan DOI, ambos paquetes se usan como archivos locales: descomprímalos en la misma carpeta y
pásela con `--data-dir`. No hay rutas fijas ni descargas automáticas.

```sh
python scripts/gmm.py verificar --data-dir /ruta/a/datos
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
