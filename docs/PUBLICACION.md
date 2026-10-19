# Estado de GitHub y preparación de Zenodo

Comprobado el 19 de octubre de 2026. Repositorio existente:
https://github.com/overtonebio/mortalidad-mexico-ia

## Estado real

GitHub está privado. Su rama principal conserva la base v13 y la entrega privada de datos del 14 de octubre.
La actualización candidata 0.2.0-rc.1 incorpora el manuscrito v20, localidad ajustada, numeración y rótulos.
El paquete local para Zenodo contiene código, los dos archivos de datos/pronósticos inalterados y la adenda
histórica. La ausencia de un DOI en esta preparación no descarta que existan borradores en otra cuenta.
No se pudo inspeccionar una cuenta de Zenodo porque la sesión disponible solicita iniciar sesión.

## Publicación por completar

1. Revisar e integrar la actualización del repositorio privado.
2. Elegir las licencias del material propio. Propuesta para decisión de los autores: MIT para código y CC BY 4.0
   para documentación y resultados propios. No se han concedido esas licencias. Los datos de terceros conservan
   sus condiciones, sin relicenciarse; la trazabilidad pendiente del extracto WPP está en DATOS.md.
3. Entrar a la cuenta de Zenodo y crear un depósito de software y otro de datos/pronósticos, con los archivos
   y los metadatos preparados. Cotejar sus licencias y reservar los DOI en los borradores si se van a citar antes
   de publicar. Un DOI reservado no implica publicación ni que sus archivos sean accesibles.
4. Publicar solo después de la aprobación de los autores. Para acceso abierto mediante GitHub-Zenodo,
   decidir además si el repositorio se hace público. El depósito manual de software permite tratar esta decisión
   por separado y debe contener la instantánea exacta del código aprobado.
5. Comprobar los DOI de versión y el acceso a los archivos, enlazar ambos registros y actualizar CITATION.cff
   y la declaración de disponibilidad del manuscrito. No usar DOI de ejemplo ni afirmar disponibilidad pública
   mediante enlaces privados.

## Consideraciones verificadas

La integración GitHub-Zenodo archiva una versión del código; no se debe asumir que incorpora los assets
externos de las entregas. Los datos y los pronósticos se preparan por separado. Las modificaciones de archivos
tras publicar están restringidas; comprobar los ZIP antes y crear una nueva versión cuando corresponda.
Si coexisten CITATION.cff y .zenodo.json, Zenodo da prioridad al segundo: por eso aquí se conserva un solo
CITATION.cff activo y los JSON de depósito se mantienen fuera del repositorio operativo.

Fuentes oficiales consultadas el 19 de octubre de 2026:
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/deposit/manage-records/
- https://help.zenodo.org/docs/deposit/manage-versions/
- https://help.zenodo.org/docs/github/describe-software/
- https://opensource.org/license/mit
- https://creativecommons.org/licenses/by/4.0/
