# Licencias: pendiente de decisión de los autores

**No se ha concedido ninguna licencia.** Mientras los autores no elijan una, este repositorio no incluye
`LICENSE` y, por tanto, no otorga permisos de reutilización más allá de lo que permita la ley. Lo siguiente es
una **propuesta** para su revisión; no es una concesión.

## Ámbitos separados

| Ámbito | Situación | Propuesta (no vigente) | Justificación |
|---|---|---|---|
| Código propio (`src/`, `scripts/`, `tests/`, `configs/`) | Sin licencia | MIT o Apache-2.0 | Licencias permisivas habituales para código científico; Apache-2.0 añade concesión explícita de patentes. Zenodo y GitHub las reconocen por su identificador SPDX. |
| Documentación (`docs/`, README) | Sin licencia | CC BY 4.0 | Permite reutilización con atribución. |
| Resultados derivados (`results/reference/`, paquetes de depósito con tablas y pronósticos) | Sin licencia | CC BY 4.0 | Compatible con los términos de libre uso del INEGI, que exigen citar la fuente y declarar las transformaciones. Requiere confirmar las condiciones de WPP 2024. |
| Datos de terceros redistribuidos (extracto INEGI, catálogos INEGI, WPP) | Condiciones de su fuente | No se relicencian | Se redistribuyen solo si sus términos lo permiten (véase `docs/DATOS.md`). |
| Figuras del artículo | Derechos según el acuerdo editorial | Decidir tras la aceptación | Los PDF finales no se incluyen; el código las regenera. |
| Pesos de modelos | No se distribuyen | — | Se obtienen de sus repositorios bajo sus propias licencias. |

## Decisiones necesarias

1. Licencia del código (y si se usa la misma para scripts históricos de entrenamiento).
2. Licencia de documentación y de resultados derivados.
3. Confirmación de las condiciones de redistribución de WPP 2024 y de los extractos INEGI.
4. Política sobre las figuras finales según la revista.

Al decidir: añadir `LICENSE` (y, si procede, `LICENSES/`), declarar el identificador SPDX en `CITATION.cff`
(campo `license`) y en los metadatos de Zenodo, y retirar este archivo.
