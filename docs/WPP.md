# WPP 2024: procedencia y reconstrucción comprobadas

Comprobado el 19 de octubre de 2026. Las **357 filas** (17 años × 21 grupos de edad) del extracto histórico
`wpp_mex_female_age5.csv` se reconstruyeron **byte por byte**, sin diferencias, desde el CSV oficial de
Naciones Unidas. Se verificaron claves únicas, cobertura completa, edades, valores positivos y finitos,
México (M49 484), mujeres, variante media y población de facto al **1 de julio**, en **miles de mujeres**.
Las hojas Excel correspondientes conservan cuatro decimales en algunos valores; el CSV oficial contiene
los tres decimales del extracto y permite una comparación exacta sin redondeos añadidos.

## Fuente y transformación

- Fuente: United Nations, Department of Economic and Social Affairs, Population Division (2024).
  *World Population Prospects 2024, Online Edition*.
- Archivo: `WPP2024_PopulationByAge5GroupSex_Medium.csv.gz`.
- [Descarga oficial](https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20%28Standard%29/CSV_FILES/WPP2024_PopulationByAge5GroupSex_Medium.csv.gz).
- [Catálogo oficial de descargas](https://population.un.org/wpp/assets/downloads.json).
- © 2024 United Nations, [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/).
- Filtros: `ISO3_code=MEX`, `LocID=484`, `VarID=2`, `Variant=Medium`, `Time=2010..2026`.
- Selección de columnas: `Time,AgeGrp,AgeGrpStart,PopFemale`, ordenadas por año y edad.
  Sin conversión de unidades, interpolación o redondeo en esta extracción. El grupo `100+` es abierto.
  El código analítico convierte miles a personas cuando construye los denominadores.
- SHA-256 fuente: `a04d7d1486a5eb2832cc812d599448f0a71e8ac9e1e7e6fa4066673d6a2487cd`.
- SHA-256 extracto y reconstrucción: `20340aaf3f3370f4c7941c85bc73996c1ca780fb5ac5fd2bce7d76f15c63e0a2`.

## Repetir la comprobación

Descargue el CSV oficial y los dos paquetes de datos del estudio. Después ejecute, desde el repositorio:

```sh
python scripts/reconstruir_wpp.py \
  --fuente-csv-gz ../WPP2024_PopulationByAge5GroupSex_Medium.csv.gz \
  --comparar-csv data/external/benchmark/datos/wpp_mex_female_age5.csv \
  --output-dir ../comprobacion_wpp
```

Solo requiere la biblioteca estándar de Python. La salida debe ser una carpeta nueva. El informe JSON
registra hashes, filtros, grano, unidades y diferencias. Una segunda copia del extracto en
`c50/paquete_original/editorial/datos/` tiene el mismo hash.
El [informe de comprobación](procedencia/wpp_20261019/verificacion_wpp.json) y el extracto reconstruido
se conservan con el código. El depósito de datos incluye la fuente oficial comprimida y este procedimiento.

Esta es una **reconstrucción independiente** que demuestra la equivalencia exacta con una fuente identificada.
No recupera la fecha ni la ruta de la descarga histórica original, cuyo registro no se conservó.
La selección del extracto es de los autores; no implica aprobación de Naciones Unidas.
