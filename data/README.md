# data/

Este directorio no contiene datos en el repositorio. Por omisión, `data/external/` (ignorado por git) es el lugar
sugerido para descomprimir los dos paquetes de depósito:

```
data/external/
  benchmark/datos/        benchmark/entradas/       (paquete «datos y entradas»)
  benchmark/pronosticos/  publicados/ y nueva_ejecucion_semillas/ (paquete «pronósticos NPZ»)
  c50/paquete_original/   c50/fuentes_inegi/        (paquete «datos y entradas»)
```

Puede usarse cualquier otra carpeta con la misma estructura mediante `--data-dir`. Antes de que exista un DOI,
los paquetes pueden usarse como archivos locales. `python scripts/gmm.py verificar --data-dir data/external`
comprueba presencia y SHA-256 de cada archivo contra `configs/datos_esperados.json`.
Catálogo de fuentes, variables y condiciones de reutilización: `docs/DATOS.md` y `docs/CATALOGO_DATOS.csv`.
