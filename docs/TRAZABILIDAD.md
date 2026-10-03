# Trazabilidad del manuscrito v13

`MATRIZ_TRAZABILIDAD.csv` relaciona cada figura (1-6, S1-S6), tabla (1-2, S1-S25 con subdivisiones) y resultado
principal con sus datos, script, configuración, salida, nivel y estado:

- **ejecutado y coincide**: ejecutado el 1-oct-2026 en entorno aislado; salida idéntica (o dentro de la tolerancia
  de redondeo indicada) a `results/reference` o al documento v13;
- **ejecutado y difiere**: ejecutado; la diferencia se describe;
- **no ejecutado**: no aplica un cálculo (tablas de texto) o es nivel 4;
- **bloqueado**: falta el generador o una entrada; el motivo se indica.

Los nombres internos no siguen la numeración del artículo (p. ej., `Figura6_C50_tendencia_y_lugar` = figura 5;
`tablaS2F_pool_calibracion.csv` = tabla S2E).
