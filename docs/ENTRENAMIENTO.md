# Nivel 4: inferencia y entrenamiento (documentado, no ejecutado)

**Estado:** el código está presente; no se ejecutó en la preparación del 1 de octubre de 2026; no hay una
reproducción comprobada de este nivel. Los pronósticos NPZ y los resultados del ajuste fino guardados son los
que sustentan el artículo.

## Requisitos

- Entorno separado de los de recálculo: `environments/entrenamiento-historico.txt` (versiones **declaradas** en
  el README fuente; existen en PyPI, pero no se instalaron ni probaron juntas en esta revisión).
  Se conservan también los registros originales recuperados en `environments/evidencia_historica/`.
- Pesos de `amazon/chronos-2`, `amazon/chronos-bolt-base` y `google/timesfm-2.5-200m-pytorch`, obtenidos de
  Hugging Face bajo sus licencias. La revisión original es desconocida; registre la que use.
- Microdatos EDR 2012-2024 completos del INEGI si se reconstruye el corpus (URL en
  `c50/fuentes_inegi/urls_datos_abiertos_inegi.json` del paquete de datos).
- Tiempo: los registros guardados (`entradas/costo_computacional.json`, tabla S7) registran, por origen, inferencia de
  18 s (Chronos-2), 12 s (Chronos-Bolt) y 134 s (TimesFM-2.5), 445 s para los cuatro clásicos en un núcleo, 56 s de
  entrenamiento de LightGBM, y 1 171 s de ajuste para la LoRA estricta (`finetune_estricto_log.json`); son mediciones de la
  ejecución original, no de esta preparación.

## Scripts y su función

| Script | Función | Salidas | Notas |
|---|---|---|---|
| `construccion_datos/build_corpus_agg.py`, `prep_series_completo.py` | Corpus y series C50 desde microdatos | `agg_mortalidad_raw.csv.gz`, series | Rutas de trabajo originales (`inegi_raw/`) |
| `construccion_datos/subregistro.py` | Clasificador de causas mal definidas | (no escribe los CSV de la tabla S5) | Etapa incompleta |
| `modelos/bench.py` | Modelos clásicos y fundacionales (v5) | `benchmark_resultados.csv.gz` | Usa `sys.path` del directorio de trabajo |
| `modelos/bench2.py`, `bench2_h48.py` | NPZ por origen (36/60 y 48 meses) | `fc_<origen>.npz`, `fc_h48_<origen>.npz` | `ONLY` limita modelos |
| `modelos/fix_clasicos.py` | Corrección de clásicos en NPZ | NPZ y `fix_clasicos_log.json` | |
| `modelos/pipeline_tsfm.py`, `pipeline_h48.py` | Aplicación a mama (contrafactual 2020-2024) | `tsfm_*.csv` | |
| `lora/finetune2.py`, `finetune_estricto.py` | Chronos-2-LoRA y ajuste completo | `finetune2_resultados.csv.gz`, `finetune_estricto_*` | Sin `lora_config`: valores por omisión de la biblioteca |

## Procedimiento sugerido (no probado)

1. Crear el entorno de entrenamiento y registrar `pip freeze`, versión de Python, CPU/GPU y revisión de pesos.
2. Ejecutar cada script en un directorio de trabajo nuevo que contenga `datos/` (copiado del paquete), nunca en
   `results/reference/`.
3. El script histórico de comparación es `src/benchmark/analisis/p5_comparar_rerun.py` (tabla S14).
   Requiere un árbol de trabajo con `codigo/`, `datos/`, `entradas/`, `pronosticos/` y `resultados/`,
   como el que crea `gmm.py benchmark`; no se ejecuta directamente desde la raíz del repositorio.
   Compara dos juegos guardados de NPZ y una comparación C50 guardada; no genera pronósticos nuevos.

Los comandos ordinarios de `scripts/gmm.py` no inician descargas ni entrenamientos.

## Comprobaciones necesarias para nuevas corridas de LoRA

Antes de entrenar, confirme que la versión elegida de Chronos detecta `peft`; detenga la ejecución si no lo detecta.
La versión 2.3.2 recurre a ajuste completo si esa dependencia falta. Para una nueva corrida, fije explícitamente
la configuración LoRA compatible con esa versión y registre rango, alfa, módulos objetivo, semillas y revisión
de los pesos. Esta recomendación no reconstruye la configuración efectiva de las corridas históricas.

Después de `fit`, compruebe que el modelo contiene la configuración PEFT y parámetros LoRA entrenables.
Conserve el adaptador, su `adapter_config.json`, versiones instaladas y stdout con el número de parámetros
entrenables y totales. `finetune2_log.json` y `finetune_estricto_log.json` registran tiempos/recuentos, no
acreditan por sí solos el modo efectivo de entrenamiento.
