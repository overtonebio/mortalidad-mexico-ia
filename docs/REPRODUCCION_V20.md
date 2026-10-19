# Reproducción del manuscrito v20

Actualización preparada el 19 de octubre de 2026. El manuscrito de referencia es el paquete de envío v20,
del 17 de octubre. La versión candidata del software es 0.2.0; no es una publicación del artículo.

## Entradas y entorno

Se usan sin alteraciones `GMM_v13_datos_y_entradas.zip` y `GMM_v13_pronosticos_NPZ.zip`, disponibles en
la [entrega de datos existente](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014).
Siga [la extracción de esos ZIP](DESCARGA_Y_EJECUCION.md), dejando `benchmark/` y `c50/` bajo `data/external/`.
El extracto C50 que necesita localidad ya está en `c50/paquete_original/editorial/datos/`.

```sh
python3.12 -m venv .venv-c50
.venv-c50/bin/python -m pip install -r environments/v20.txt
.venv-c50/bin/python scripts/gmm_v20.py verificar --data-dir data/external
```

El entorno verificado de C50 tiene Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 y Matplotlib 3.11.2.
Se conservan por separado los requisitos históricos del benchmark. La comparación exacta de imágenes usa
Arial y las versiones indicadas; en otra plataforma pueden cambiar las fuentes sin cambiar los valores.

## Doce figuras vigentes

```sh
.venv-c50/bin/python scripts/gmm_v20.py figuras --data-dir data/external --output-dir ../ejecuciones/figuras_v20
```

La salida `figuras/` contiene Figura_1–Figura_6 y Figura_S1–Figura_S6 en PDF. Primero se reconstruye la base
v13 en una copia aislada; después se aplican los generadores vigentes de 3, 5, 6, S1, S2, S3 y S5. Las pruebas
internas comparan los artistas numéricos antes y después de los cambios de texto. S2 comprueba además el
hash de su imagen de control v19, sin necesitar distribuir esa imagen.

Los doce PDF regenerados coincidieron píxel a píxel a 150 ppp y en todo el texto extraído con los del envío
v20. Sus bytes PDF difieren por metadatos de generación; `configs/figuras_v20.json` identifica los PDF de envío.
La comprobación se registra en `docs/validacion_v20/figuras.json`. Es regeneración desde resultados guardados,
no entrenamiento ni recálculo integral de todos los análisis.

## Ampliación ajustada por localidad

```sh
.venv-c50/bin/python scripts/gmm_v20.py localidad --data-dir data/external --output-dir ../ejecuciones/localidad_v20
```

La ejecución copia los scripts y las entradas en un directorio nuevo; conserva el plan del 5 de octubre y todas
las variantes previstas. Ejecuta el ajuste, las verificaciones independientes y una comparación automática
contra `results/reference/v20/localidad/`. Los 20 CSV comparados coincidieron byte a byte con las referencias.
Las comprobaciones incluyen ajuste Bernoulli individual frente al binomial agrupado y derivadas por diferencias
finitas. No se cambian resultados de referencia ni entradas. `--prepare-only` prepara las copias sin ajustar.

Las rutas largas dentro de `trabajo/` mantienen la estructura que esperan los scripts históricos; se crean bajo
la salida elegida y no hacen referencia a carpetas personales. Los nombres históricos tabla_S26A/B/C equivalen
a las tablas S13/S14/S15 del suplemento v20. La correspondencia completa está en
`configs/Mapa_numeracion_v15_v16.csv`.

## Base científica y límites

El benchmark y los análisis C50 previos conservan sus comandos `scripts/gmm.py benchmark` y `c50`. No se
reentrenaron modelos durante esta actualización. Los pesos y los entornos de entrenamiento se describen en
[ENTRENAMIENTO.md](ENTRENAMIENTO.md). Las limitaciones de procedencia documentadas en
[DATOS.md](DATOS.md) y [REPRODUCCION.md](REPRODUCCION.md) siguen vigentes.
