# Reproducción por niveles: qué se ejecutó y qué se comprobó

Preparación del 1 de octubre de 2026. Plataforma de prueba: macOS arm64, 11 núcleos lógicos, 18 GB, sin GPU.
Los registros completos de esta comprobación están en `revision/VALIDACION.md` de la entrega (fuera del
repositorio público).

## Distinciones

- **Código presente**: el script existe en `src/`.
- **Comando ejecutable**: `scripts/gmm.py` lo ejecuta con entradas verificadas.
- **Reproducción comprobada**: se ejecutó en un entorno aislado y su salida se comparó con `results/reference`.

## Nivel 1 — integridad y formato

`gmm.py verificar` comprobó 113/113 archivos de referencia, 109/109 entradas, 108 matrices NPZ y el corpus.
`gmm.py verificar-archivo` comprobó los 292 archivos del manifiesto original del archivo científico v13. Esto
demuestra que los archivos son los mismos; **no** valida resultados.

## Nivel 2 — figuras desde resultados guardados

`gmm.py figuras` regeneró las 12 figuras. Al rasterizar a 150 y 300 ppp, los PDF regenerados son idénticos píxel
a píxel a los del envío v13 y su texto extraído es idéntico; los bytes difieren porque cada PDF registra su fecha
de creación. Resultado igual partiendo de `results/reference` y de las salidas de nivel 3. Entorno recomendado:
`environments/c50-figuras.txt` (numpy 2.3.5, pandas 2.2.3, matplotlib 3.11.2), con el que también coincide la
huella de datos de los artistas de las figuras 6 y S3 (`results/reference/figuras/validacion_datos_figuras.json`).
Con el entorno del benchmark (numpy 2.4.6, pandas 2.3.3) el render es idéntico, pero la huella de la figura 6
difiere. La revisión de Codex identificó dos coordenadas x de `patch_extents` en el tercer panel:
la diferencia máxima es 2.2737367544323206e-13 en coordenadas de dibujo. Los valores de las barras,
líneas, límites y demás artistas coinciden; los PDF siguen siendo idénticos al rasterizarlos.

## Nivel 3 — recálculo

| Módulo | Entorno probado | Tiempo | Resultado de `gmm.py comparar` |
|---|---|---|---|
| benchmark (12 pasos originales + `tabla1_y_S15.py`) | Python 3.11.16; `benchmark-recalculo.txt` (más matplotlib 3.11.2 instalado) | 48-53 s | 62 archivos idénticos en bytes (incluida `tablaS15`); 4 `.csv.gz` con contenido idéntico (cambia la cabecera gzip); `numeros_v9.json` dentro de tolerancia (máx. dif. absoluta 2.2e-14 en 3 717 valores); `tablaS11C_parametros_cosecha.csv` difiere en un texto bibliográfico (diferencia conocida); `tabla1_manuscrito.csv` es nueva (sin referencia) |
| c50 (11 pasos, NSIM = 10 000, 2 hilos) | Python 3.12.14; `c50-recalculo.txt` + matplotlib | 236-241 s (227-231 s en `lugar_defuncion`) | 41 archivos idénticos en bytes; `verificacion_v11.json` difiere solo por las claves del cotejo opcional con v10 (`--baseline`, paquete no incluido); `v12_apvp_localidad/validacion.json` no se regenera (documentado) |

Las diferencias analizadas están en `configs/diferencias_conocidas.json`; `comparar` solo las acepta si verifica la sustitución bibliográfica exacta, la omisión de tres claves
de cotejo histórico, el hash de la tabla nueva o la ausencia documentada del informe histórico.
Un cambio de cifras en esos mismos archivos falla. También falla si la carpeta está vacía, faltan figuras,
aparecen salidas inesperadas o una diferencia numérica contiene NaN unilateral. Tolerancia: igualdad exacta primero; para números, |a − b| ≤ 1e-12 + 1e-9·|b| (redondeo de doble precisión). No
se ampliaron tolerancias.

Además, la tabla 1 regenerada coincide celda por celda con la del Word v13; las celdas de datos de la tabla 2 son
idénticas a `tabla2_v9.csv`; y las cifras impresas de las 41 tablas suplementarias con datos se encontraron en sus
CSV/JSON de origen (cotejo descrito en `MATRIZ_TRAZABILIDAD.csv`).

## Nivel 4 — inferencia y entrenamiento

No ejecutado (véase `ENTRENAMIENTO.md`).

## Separación de análisis

- Detección (figura 3, tablas S9-S9D): umbral calibrado con el origen 2014, ajuste δ con 2015 y prueba con 2016
  (años 2017-2019).
- Incertidumbre predictiva (tabla 2, S2B-S2G, figura 4): intervalos conformales entre series.
- Análisis exploratorios C50 (figura 5, S4-S6, tablas S17-S25): IC de parámetros, sin carácter confirmatorio.

## Advertencias técnicas encontradas

- Los scripts C50 importan módulos hermanos (`comun.py`) desde su directorio. Si `PYTHONSAFEPATH=1` o
  `python -P`, `verificar_correcciones_v11.py` falla; `gmm.py` retira esa variable en los procesos hijos.
- `figuras_c50.py` necesita `resultados/c50_analitico.pkl`, que no se distribuye; `gmm.py figuras` lo reconstruye
  con `validacion_datos.py` en una copia aparte.

## Revisión adicional de Codex (1 de octubre de 2026)

Se repitieron el benchmark (13 pasos), C50 (11 pasos; NSIM 10 000 y dos hilos) y la generación de
las 12 figuras desde las salidas recién calculadas. Las comparaciones estrictas pasan; las 12 figuras
coinciden píxel a píxel y en texto con la v13 a 150 y 300 ppp. La tabla 1 coincide en sus 77 celdas,
incluidos encabezados, con el Word v13. Un cálculo independiente de WQL desde los NPZ confirma
reducciones de 13.3350 % (LightGBM) y 12.4329 % (ensamble).

Se añadieron pruebas de regresión del comparador. Los datos, NPZ, scripts históricos de análisis y
resultados de referencia se conservaron. El entrenamiento y las etapas sin generador recuperado
siguen sin reproducción comprobada. Registros completos: carpeta interna `revision/codex/` de la entrega.

## Reconstrucción adicional de S1 y S2A

```sh
.venv-benchmark/bin/python src/benchmark/analisis/reconstruir_S1_S2A.py \
  --data-dir data/external --output-dir ejecuciones/S1_S2A
```

Este script nuevo reconstruye los 120 valores numéricos de S2A desde los NPZ de 48 meses (todos
coinciden exactamente) y los 140 conteos de S1 desde el corpus de 2 870 series. Conserva Tipo/CIE10
como definiciones de entrada. El generador histórico no se recuperó; tampoco se regenera el corpus
o los pronósticos. Guarda los CSV nuevos y `validacion_S1_S2A.json` en una carpeta nueva.
