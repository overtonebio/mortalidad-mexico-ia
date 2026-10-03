# Entornos

Se mantienen separados porque el benchmark y C50 se reprodujeron con versiones incompatibles de numpy/pandas.

| Archivo | Uso | Tipo |
|---|---|---|
| `benchmark-recalculo.txt` | Nivel 3, benchmark (Python 3.11) | Reconstruido con las versiones del entorno fuente; **probado** el 1-oct-2026 (Python 3.11.16, macOS arm64) |
| `benchmark-figuras.txt` | Figuras con el entorno del benchmark | Probado: render idéntico; la huella de datos de la figura 6 difiere |
| `c50-recalculo.txt` | Nivel 3, C50 (Python 3.12) | Reconstruido con las versiones de la reproducción previa; **probado** (Python 3.12.14, macOS arm64) |
| `c50-figuras.txt` | Nivel 2 (recomendado para las 12 figuras) | Probado: render y huella de datos idénticos a la referencia |
| `comparacion-figuras.txt` | Comparación por render | Probado con pypdfium2 5.9.0 |
| `entrenamiento-historico.txt` | Nivel 4 | **Histórico declarado**: versiones existentes en PyPI, no instaladas ni probadas |
| `historico/` | Archivos originales del archivo científico v13 | Sin cambios |

«Probado» significa: instalado en un entorno aislado nuevo y usado para ejecutar los pasos con salidas comparadas
con la referencia. No es un `pip freeze` completo: solo se fijan las dependencias directas; las transitivas las
resolvió pip el 1 de octubre de 2026 (véanse los registros en la entrega).
