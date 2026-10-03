# Documentación técnica

## Arquitectura

`scripts/gmm.py` no reescribe los scripts científicos. Para cada ejecución:

1. verifica las entradas de `--data-dir` contra `configs/datos_esperados.json` (SHA-256 y tamaño);
2. crea `--output-dir` (debe ser nuevo o estar vacío; no puede estar en `results/reference` ni contener el
   repositorio) y copia ahí el código de `src/` y las entradas, con la misma estructura relativa que esperaban
   los scripts originales (`codigo/`, `datos/`, `entradas/`, `pronosticos/`, `resultados/` …);
3. ejecuta los pasos de `configs/pasos.json`, en el orden de los ejecutores originales, con
   `sys.executable`, variables de entorno explícitas (`ESTRATOS`, `NSIM`, `OPENBLAS_NUM_THREADS`,
   `OMP_NUM_THREADS`) y un registro por paso en `logs/`;
4. comprueba que las entradas copiadas no cambiaron y escribe `registro_ejecucion.json`.

`gmm.py comparar` compara las salidas con `results/reference` (`scripts/comparar_resultados.py`) y, si se
indica `--figuras-v13`, las figuras por render (`scripts/comparar_pdf.py`).

## Mapa del código

| Ruta | Procedencia | Notas |
|---|---|---|
| `src/benchmark/auditoria/reproducir_benchmark.py` | archivo v13 `benchmark/codigo/` | `python reproducir_benchmark.py <dir_npz> <dir_salida>` |
| `src/benchmark/auditoria/limpiar_tablas.py` | recuperado (`paquete_GMM_SI_IA_v10_revision`) | requerido por `tabla1_y_S15.py` |
| `src/benchmark/analisis/*.py` | archivo v13 | sufijos `v9` = procedencia, no versión del manuscrito |
| `src/benchmark/analisis/tabla1_y_S15.py` | **nuevo**: extracción de `generar_manuscrito.py` | genera tabla 1 y S15 sin Word |
| `src/benchmark/documentos/referencias.py` | archivo v13 | lista bibliográfica que importa `tablas_v9.py`; no genera documentos |
| `src/benchmark/{modelos,lora,construccion_datos}/` | archivo v13 | nivel 4; rutas de trabajo originales |
| `src/c50/*.py` | archivo v13 `c50/codigo/` | `comun.py` fija el directorio de trabajo en `codigo/` |
| `src/c50/descargar_catalogos.py`, `listar_miembros_zip.py`, `zip_remoto.py` | archivo v13 | requieren red; **no** forman parte de los pasos ordinarios |
| `src/figuras/figuras_base_v9.py`, `figs_v4.py`, `estilo_figuras.py` | archivo v13 `editorial_v13/codigo/` | funciones de figuras |
| `src/figuras/actualizar_figuras.py` | archivo v13 | figuras 6 y S3 con rótulos v13; verifica que no cambian los datos de los artistas |
| `src/figuras/exportar_figuras_v13.py` | **adaptado** de `exportar_figuras_pdf.py` recuperado | figuras 2, 3, 4, S1, S2 |
| `src/figuras/figura_diseno_v11.py` | recuperado (`paquete_C50_v11_corregido`) | figura 1 |

Los nombres de archivo internos no indican la numeración del artículo. Correspondencias:
`configs/figuras.json`; tablas: `docs/MATRIZ_TRAZABILIDAD.csv` (por ejemplo, el CSV interno
`tablaS2F_pool_calibracion.csv` es la tabla S2E del artículo y `tablaS2G_comprobacion_separada.csv`, la S2F).

## Semillas y determinismo

- Benchmark de recálculo: bootstrap y simulaciones con semillas fijas en cada script; salidas idénticas.
- C50: `lugar_defuncion.py` usa `SEED = 20260930` y `NSIM` (10 000 por omisión); la referencia se generó con
  dos hilos BLAS/OpenMP.
- Pronósticos y entrenamiento: véase `MODELOS.md`.

## Pruebas

`python -m unittest discover -s tests -v`: ayuda de la CLI, protección de `results/reference`, errores
informativos, configuración, comparador (datos sintéticos), detección de formas y no finitos en NPZ
(sintéticos), integridad de la referencia y ausencia de rutas personales. La integración continua
(`.github/workflows/pruebas.yml`) ejecuta lo mismo sin credenciales, GPU ni descargas.
