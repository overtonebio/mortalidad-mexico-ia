# Estrategia de publicación en GitHub y archivo en Zenodo (propuesta)

## Distribución privada para revisión

El repositorio privado `overtonebio/mortalidad-mexico-ia` se creó el 3 de octubre de 2026. El 14 de octubre se añadió la [entrega privada `datos-v13-20261014`](https://github.com/overtonebio/mortalidad-mexico-ia/releases/tag/datos-v13-20261014) con los dos ZIP originales v13, sus hashes e instrucciones para los colaboradores. La [guía de descarga y ejecución](DESCARGA_Y_EJECUCION.md) describe su uso.

Esta distribución mantiene el acceso privado del repositorio. La publicación pública del software y de los datos, el depósito en Zenodo y los DOI siguen pendientes. Las propuestas de los apartados siguientes se refieren a esa publicación futura; la etiqueta de distribución privada de datos no es una versión pública del software.

## Fuentes consultadas (1 de octubre de 2026)

- GitHub Docs, *About CITATION files*
  (https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files):
  `CITATION.cff` en la raíz del repositorio; GitHub ofrece la cita en APA y BibTeX; `preferred-citation` permite
  dirigir la cita a un artículo. Página descargada y leída.
- Zenodo Help, *Describe software* (https://help.zenodo.org/docs/github/describe-software/): Zenodo acepta
  `CITATION.cff` y `.zenodo.json` para releases de GitHub; **si ambos existen, solo usa `.zenodo.json`** e ignora
  `CITATION.cff`.
- Zenodo Help, *Manage files* y *Create new upload* (https://help.zenodo.org/docs/deposit/manage-files/,
  https://help.zenodo.org/docs/deposit/create-new-upload/): hasta 100 archivos y 50 GB por registro; recomiendan
  empaquetar en ZIP cuando hay más de 20 archivos y retirar archivos ocultos (`.DS_Store`, `__MACOSX`); tras publicar,
  los archivos solo pueden modificarse por el depositante durante 45 días.
- Zenodo FAQ, *How to specify a license for a software record on GitHub*: prioridad de licencia
  `.zenodo.json` > `CITATION.cff` > licencia del repositorio > licencia por omisión de Zenodo.
- Esquema oficial CFF 1.2.0 (https://citation-file-format.github.io/1.2.0/schema.json): `CITATION.cff` de este
  repositorio validado contra él.

En la revisión de Codex del 1 de octubre de 2026 se leyeron completas las guías oficiales de
[archivo desde GitHub](https://help.zenodo.org/docs/github/archive-software/github-upload/),
[metadatos JSON](https://help.zenodo.org/docs/github/describe-software/zenodo-json/) y
[versiones](https://help.zenodo.org/docs/deposit/manage-versions/). El problema de acceso informado en la
preparación original ya no impide comprobar esas instrucciones.

Los JSON en `zenodo/metadatos_borrador/` de la entrega contienen `_estado` y `_pendientes` como notas internas.
Antes de usar un borrador como `.zenodo.json` o con una API, retire esas claves, complete licencia/versión reales
y valide los campos contra el esquema de la interfaz elegida. No son archivos directamente publicables.

## Estrategia propuesta

1. **Código (GitHub → Zenodo).** Para la futura publicación, usar el repositorio `mortalidad-mexico-ia` ya creado y, tras decidir su publicación pública, activar la integración GitHub-Zenodo para ese repositorio y publicar un release etiquetado (por ejemplo
   `v0.1.0`). Zenodo archiva el código fuente de esa etiqueta y le asigna un DOI de versión. No asuma que la
   integración incorpora los paquetes de datos, los NPZ ni otros archivos externos al repositorio: no están en git
   y deben depositarse aparte.
2. **Metadatos del software.** Mantener solo `CITATION.cff` mientras basten sus campos (autores, ORCID,
   afiliaciones, título, versión, palabras clave). Añadir `.zenodo.json` solo si se necesitan campos propios de
   Zenodo (comunidades, identificadores relacionados, financiamiento); en ese caso, **Zenodo ignorará
   `CITATION.cff`**, y ambos archivos deben mantenerse coherentes. Borrador no activo en
   `zenodo/metadatos_borrador/` de la entrega.
3. **Datos y pronósticos (depósito manual en Zenodo).** Dos registros de tipo *dataset* (o un registro con dos
   ZIP): «datos y entradas» y «pronósticos NPZ». Cada ZIP lleva su `LEEME` y `MANIFIESTO_SHA256.json`. Con los
   límites citados (100 archivos, 50 GB) ambos caben holgadamente. Confirmar antes las condiciones de WPP 2024
   y la licencia de los derivados.
4. **Enlaces cuando existan identificadores reales.** En el registro de software, *related identifiers*:
   «is supplemented by» → DOI de los datos; «is cited by»/«is supplement to» → DOI del artículo. En el registro de
   datos, la relación inversa. En `CITATION.cff`, añadir `doi` (del software) y `preferred-citation` (artículo)
   solo cuando existan. En el artículo, citar el DOI de **versión** usado.
5. **DOI de versión y DOI de concepto.** Cada versión publicada en Zenodo tiene su propio DOI; el registro agrupa
   las versiones bajo un DOI de concepto que remite a la más reciente. Para reproducibilidad, cite en el artículo
   el DOI de versión; use el de concepto solo para referirse al proyecto en general. (Comportamiento descrito en
   la ayuda de Zenodo sobre versiones; verificar en https://help.zenodo.org/docs/deposit/manage-versions/.)
6. **Antes de una versión pública.** Elegir licencias (`LICENSING_PENDING.md`), actualizar `version` en `CITATION.cff`,
   ejecutar `python -m unittest discover -s tests`, `gmm.py verificar` y revisar que no haya rutas personales ni
   secretos.

## Lista de comprobación

- [ ] Licencia del código, documentación y derivados decidida y añadida (`LICENSE`, SPDX en metadatos).
- [ ] Condiciones de WPP 2024 confirmadas.
- [ ] Decisión sobre la redistribución de figuras finales según la revista.
- [x] Repositorio privado creado; paquetes v13 distribuidos en una entrega privada para revisión.
- [ ] Versión pública del software publicada y DOI de versión recibido.
- [ ] Depósitos de datos y pronósticos publicados; DOI recibidos.
- [ ] `CITATION.cff` (y, si se usa, `.zenodo.json`) actualizados con DOI, URL y fecha reales.
- [ ] Enlaces cruzados entre software, datos y artículo.
