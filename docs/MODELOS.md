# Modelos: nombres, configuración y procedencia

Los nombres siguen el manuscrito v13 (métodos S13). Las etiquetas internas de los archivos se conservan.

| Nombre en el artículo | Etiqueta interna en NPZ/CSV | Tipo | Entrenamiento | Código |
|---|---|---|---|---|
| Ingenuo estacional | `Ingenuo estacional` | Clásico | Por serie | `src/benchmark/modelos/bench.py` |
| Log-lineal | `Log-lineal` | Clásico (tendencia + mes) | Por serie | `bench.py`, `pipeline_tsfm.py` |
| SARIMA | `SARIMA` | Clásico, (1,1,1)(0,1,1)12 en log | Por serie | `bench.py`, `pipeline_tsfm.py` |
| ETS | `ETS amortiguado` | Clásico, aditivo amortiguado | Por serie | `bench.py`, `pipeline_tsfm.py` |
| LightGBM | `LightGBM global` | Global, entrenado con las series del estudio | Desde cero por origen | `bench2.py` |
| N-HiTS | `N-HiTS global` | Global, entrenado con las series del estudio | Desde cero por origen | `bench2.py`, `bench2_h48.py` |
| Chronos-2 | `Chronos-2` | Fundacional preentrenado (`amazon/chronos-2`) | Sin adaptación | `bench.py`, `pipeline_tsfm.py` |
| Chronos-Bolt | `Chronos-Bolt` | Fundacional preentrenado (`amazon/chronos-bolt-base`) | Sin adaptación | ídem |
| TimesFM-2.5 | `TimesFM-2.5` | Fundacional preentrenado (`google/timesfm-2.5-200m-pytorch`) | Sin adaptación | ídem |
| Ensamble | `Ensamble (mediana)` | Mediana de Log-lineal, SARIMA, ETS, Chronos-2, Chronos-Bolt y TimesFM-2.5 | Se calcula desde los miembros | `reproducir_benchmark.py` |
| Chronos-2-LoRA | `LoRA 1000 s0`, `LoRA 1000 s1`, `LoRA 1000 s2`, `LoRA 300 s0`, `LoRA 1000 sin mama`, `LoRA 1000 estricto …` | Adaptación de Chronos-2 mediante LoRA | Ajuste fino | `src/benchmark/lora/` |
| Chronos-2, ajuste completo | `Completo 1000 s0` | Actualización de todos los parámetros | Ajuste fino | `lora/finetune2.py` |

LightGBM y N-HiTS son arquitecturas publicadas, entrenadas para este estudio; Chronos-2-LoRA no es una
arquitectura nueva. El clasificador auxiliar de causas mal definidas (LightGBM; tabla S5) es otro modelo.

## Configuraciones registradas en el código

- **LightGBM** (`bench2.py`): cuantiles 0.1, 0.5 y 0.9 por separado; 400 iteraciones; tasa 0.05; máximo de 63 hojas;
  mínimo 100 observaciones por hoja; fracción de variables 0.9; `seed=0` (por origen); `num_threads=10`.
- **N-HiTS** (`bench2.py`): 1 000 pasos; lotes de 64; ventanas de 256; escalamiento robusto; `random_seed=1`;
  entrada de 24 meses si la historia es ≥ 60 meses y de 12 en otro caso.
- **Clásicos**: `bench.py` usa `rng=np.random.default_rng(i)` por serie (500 simulaciones); `pipeline_tsfm.py`
  y `pipeline_h48.py` usan 2 000 simulaciones y `torch.manual_seed(0)`.
- **Hilos**: `torch.set_num_threads(10)` en `bench*.py` y `lora/*.py`; 8 en `pipeline_*.py`. CPU
  (`device_map='cpu'`); no se registró uso de GPU.
- **Chronos-2-LoRA** (`lora/finetune2.py`, `lora/finetune_estricto.py`): origen dic-2016; datos 2012-2016;
  evaluación 2017-2019 (36 meses); lotes de 64; horizonte de entrenamiento 12; contexto 60; historia mínima 24;
  tasa 1e-5 (LoRA) y 1e-6 (ajuste completo); 1 000 pasos (300 en una variante); semillas 0, 1 y 2 en la variante
  principal. Exclusiones: «sin mama» retira las 66 series `C. mama`; «estricta» retira además las 264 de
  `Todas las causas` (330 excluidas, 2 540 de entrenamiento; `entradas/finetune_estricto_log.json`).
  Recuentos comprobados en `corpus2_mortalidad_mensual.csv.gz`.
- **Rango y alfa de LoRA**: los scripts **no** los fijan (no pasan `lora_config`). En el código fuente de
  `chronos-forecasting` 2.3.2 (versión declarada; rueda `chronos_forecasting-2.3.2-py3-none-any.whl`,
  SHA-256 `0f0d9a1972f252d6cf584b9fa749bd1b389b3c1cf05a889c4130cb10652c2117`, inspeccionada el 1 de octubre de
  2026), el valor por omisión es `r=8`, `lora_alpha=16`, módulos `self_attention.{q,k,v,o}` y
  `output_patch_embedding.output_layer`. Esto describe la biblioteca, **no** un registro de la ejecución
  original; además, esa versión vuelve a ajuste completo con una advertencia si `peft` no está instalado.
  Los registros históricos recuperados (`environments/evidencia_historica/`) incluyen
  `chronos-forecasting==2.3.2` y `peft==0.21.0`, y coinciden con los de la versión de trabajo v8.
  Respaldan el entorno declarado, pero no identifican inequívocamente el entorno de cada corrida.
  No se conservaron adaptadores ni registros de parámetros entrenables por corrida. No hay evidencia de que
  hubiera ocurrido el cambio a ajuste completo; la trazabilidad específica sigue incompleta.
  Véase `ENTRENAMIENTO.md` para las comprobaciones de futuras ejecuciones.
- **Figura S3**: piloto histórico (1 328 series, dos orígenes) con variantes de contexto entre series y LoRA;
  su configuración no se documenta en estos scripts y no debe atribuírsele la de la figura 6.

## Pesos y revisiones

| Modelo | Repositorio | Revisión usada originalmente |
|---|---|---|
| Chronos-2 | `amazon/chronos-2` | **Desconocida** (no registrada) |
| Chronos-Bolt | `amazon/chronos-bolt-base` | **Desconocida** |
| TimesFM-2.5 | `google/timesfm-2.5-200m-pytorch` | **Desconocida** |

No se distribuyen pesos. Una descarga actual no debe presentarse como la revisión del análisis.
