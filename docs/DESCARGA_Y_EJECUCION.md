# Descargar los datos y ejecutar los análisis

Actualizado el 19 de octubre de 2026. Los dos paquetes originales v13 están disponibles como archivos adjuntos de la [entrega de datos `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014). Los mismos ZIP se archivan en el [depósito de datos](https://doi.org/10.5281/zenodo.23480195).

Esta entrega corresponde al código y los resultados de referencia v13 conservados aquí. Las versiones editoriales posteriores del manuscrito pueden cambiar numeración y redacción; esta carga no reemplaza los análisis ni afirma una reproducción integral de esas versiones.

La actualización v20 usa estos mismos archivos sin cambios. Para generar las figuras vigentes y recalcular
localidad después de extraerlos, siga [REPRODUCCION_V20.md](REPRODUCCION_V20.md); los comandos históricos
de abajo siguen disponibles para la base v13.

## 1. Descargar los dos paquetes

Abra la [entrega de datos](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014), despliegue **Assets** y descargue:

| Archivo | Tamaño | Contenido |
|---|---:|---|
| `GMM_v13_datos_y_entradas.zip` | 7 958 985 bytes (7.96 MB) | Corpus, entradas y extractos C50 |
| `GMM_v13_pronosticos_NPZ.zip` | 135 286 834 bytes (135.29 MB) | Los 12 archivos de pronósticos NPZ |
| `SHA256SUMS.txt` | Texto | Hashes de los dos ZIP |

**Source code (zip)** y **Source code (tar.gz)** contienen el código de la etiqueta; los datos se descargan mediante los dos archivos con nombre `GMM_v13_...`.

Coloque los dos ZIP y `SHA256SUMS.txt` dentro de `descargas/`, en la raíz de su copia local del repositorio. La estructura inicial debe ser:

```text
mortalidad-mexico-ia/
  scripts/gmm.py
  configs/
  descargas/
    GMM_v13_datos_y_entradas.zip
    GMM_v13_pronosticos_NPZ.zip
    SHA256SUMS.txt
```

Alternativa con GitHub CLI, si ya está instalada y autenticada (`gh auth login`):

```sh
gh release download datos-v13-20261014 --repo overtonebio/mortalidad-mexico-ia --pattern 'GMM_v13_*.zip' --pattern SHA256SUMS.txt --dir descargas
```

Los archivos están disponibles en la entrega de datos de GitHub y en Zenodo.

## 2. Comprobar y descomprimir

Ejecute los comandos desde la raíz `mortalidad-mexico-ia`. Para comprobar los ZIP en Linux:

```sh
(cd descargas && sha256sum -c SHA256SUMS.txt)
```

En macOS:

```sh
(cd descargas && shasum -a 256 -c SHA256SUMS.txt)
```

Ambos archivos deben indicar `OK`. Los mismos hashes están en [data/SHA256SUMS_paquetes_v13.txt](../data/SHA256SUMS_paquetes_v13.txt), para cotejar el manifiesto descargado con el repositorio.

```sh
mkdir -p data/external
unzip -o descargas/GMM_v13_datos_y_entradas.zip 'benchmark/*' 'c50/*' -d data/external
unzip -o descargas/GMM_v13_pronosticos_NPZ.zip 'benchmark/*' -d data/external
```

Estas selecciones extraen las entradas analíticas. Los metadatos originales quedan dentro de cada ZIP. `-o` permite repetir la extracción de estos mismos paquetes, sustituyendo su copia local de las entradas. La carpeta final contiene:

```text
data/external/
  benchmark/
    datos/
    entradas/
    pronosticos/
      publicados/
      nueva_ejecucion_semillas/
  c50/
    paquete_original/
    fuentes_inegi/
```

## 3. Preparar los entornos

Si ya instaló los entornos indicados en el README, continúe con el paso 4. Se requieren Python 3.11 para el benchmark y Python 3.12 para C50 y figuras, además de `pip` y `unzip`:

```sh
python3.11 -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r environments/benchmark-recalculo.txt
python3.12 -m venv .venv-c50
.venv-c50/bin/python -m pip install -r environments/c50-figuras.txt
```

Los comandos siguientes son para Linux/macOS y usan explícitamente el entorno apropiado, aunque tenga otro entorno activado.

## 4. Verificar las entradas

```sh
.venv-benchmark/bin/python scripts/gmm.py verificar --data-dir data/external
```

Si tiene activado `.venv-benchmark`, también puede usar `python scripts/gmm.py verificar --data-dir data/external`.

Salida esperada:

```text
Resultados de referencia: 113/113 archivos íntegros
Entradas en --data-dir: 109/109 archivos íntegros
NPZ: 108 matrices revisadas (9 modelos × 12 archivos esperados = 108)
Corpus: 2 870 series × 156 meses (2012-2024), finitos y no negativos.
PASS
```

La comprobación del 14 de octubre de 2026 pasó con los dos paquetes originales. Este nivel verifica integridad y formato.

## 5. Ejecutar los scripts

Use una carpeta de salida nueva o vacía para cada ejecución.

Benchmark desde pronósticos guardados:

```sh
.venv-benchmark/bin/python scripts/gmm.py benchmark --data-dir data/external --output-dir ../ejecuciones/benchmark_revision
.venv-benchmark/bin/python scripts/gmm.py comparar --output-dir ../ejecuciones/benchmark_revision
```

Análisis de cáncer de mama y sus figuras:

```sh
.venv-c50/bin/python scripts/gmm.py c50 --data-dir data/external --output-dir ../ejecuciones/c50_revision --figuras
.venv-c50/bin/python scripts/gmm.py comparar --output-dir ../ejecuciones/c50_revision
```

Las 12 figuras desde resultados de referencia guardados:

```sh
.venv-c50/bin/python scripts/gmm.py figuras --data-dir data/external --output-dir ../ejecuciones/figuras_revision
```

El benchmark recalcula métricas desde los NPZ. C50 repite los análisis exploratorios con las entradas distribuidas. El reentrenamiento y la generación original de pronósticos se describen por separado en [ENTRENAMIENTO.md](ENTRENAMIENTO.md). Alcance y limitaciones: [REPRODUCCION.md](REPRODUCCION.md).

## Errores frecuentes

- **`--data-dir no existe`**: ejecute desde la raíz del repositorio y confirme que `data/external/` contiene directamente `benchmark/` y `c50/`.
- **`unrecognized arguments: --archivo`**: para estos dos paquetes use `verificar --data-dir data/external`.
- **`verificar-archivo` pide `--archivo`**: ese subcomando verifica la carpeta descomprimida `Analisis_y_pronosticos` del archivo científico completo. Para la entrega de dos ZIP siga el paso 4.
- **`--output-dir ya contiene archivos`**: elija otro nombre para la ejecución; los resultados de referencia se conservan.
- **`ModuleNotFoundError`**: instale los requisitos y use el ejecutable del entorno indicado para ese análisis.
- **Página 404 al abrir la entrega**: utilice el DOI de datos indicado arriba y compruebe el nombre del archivo.
