# Registros históricos recuperados

Los archivos fuente `codigo/modelos/requirements_tsfm.txt` y `versiones_tsfm.txt` de la carpeta original
coinciden byte a byte con los de la versión v8. Sus SHA-256 originales son:

- requirements_tsfm.txt: e4745efdf9143b2e11b99266a9a3447a33f81b81ce9372db5cfdc56e0a595a29
- versiones_tsfm.txt: 2aa5601aa0d4fec6632445982a48b3193e1acca946edb9244fc0d3fe11bca72e

La copia pública de `requirements_tsfm.txt` omite las rutas locales de compilación con `file://`; no
infiere versiones para esas líneas. `versiones_tsfm.txt` se conserva sin cambios. Incluyen
chronos-forecasting 2.3.2 y peft 0.21.0. Los originales se conservan en la revisión interna.

Respaldan el entorno registrado, pero no certifican que cada corrida se ejecutara en él. No son
los entornos de recálculo instalados y comprobados ni una receta portable de instalación.
