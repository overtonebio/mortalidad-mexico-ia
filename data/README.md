# Datos externos

Los dos paquetes v13 están disponibles en **Assets** de la [entrega privada `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014). Se requiere una cuenta de GitHub con acceso al repositorio.

Siga la [guía de descarga y ejecución](../docs/DESCARGA_Y_EJECUCION.md). Coloque los ZIP descargados en `descargas/` y ejecute desde la raíz del repositorio:

```sh
mkdir -p data/external
unzip -o descargas/GMM_v13_datos_y_entradas.zip 'benchmark/*' 'c50/*' -d data/external
unzip -o descargas/GMM_v13_pronosticos_NPZ.zip 'benchmark/*' -d data/external
.venv-benchmark/bin/python scripts/gmm.py verificar --data-dir data/external
```

Estructura esperada:

```text
data/external/
  benchmark/datos/
  benchmark/entradas/
  benchmark/pronosticos/publicados/
  benchmark/pronosticos/nueva_ejecucion_semillas/
  c50/paquete_original/
  c50/fuentes_inegi/
```

`data/external/` permanece ignorada por Git. Puede emplearse otra carpeta con la misma estructura mediante `--data-dir`. Los hashes de los ZIP están en [SHA256SUMS_paquetes_v13.txt](SHA256SUMS_paquetes_v13.txt); los de las 109 entradas, en `configs/datos_esperados.json`.

Fuentes y condiciones de reutilización: [docs/DATOS.md](../docs/DATOS.md) y [docs/CATALOGO_DATOS.csv](../docs/CATALOGO_DATOS.csv).
