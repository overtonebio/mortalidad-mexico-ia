# Análisis y pronósticos de apoyo

Manuscrito: SI: Inteligencia artificial para estimar la mortalidad esperada en México: evaluación comparativa y aplicación al cáncer de mama

English title: SI: Artificial intelligence for estimating expected mortality in Mexico: comparative evaluation and application to breast cancer

Preparación del archivo: 1 de octubre de 2026. Los nombres v9, v11 y v12 identifican
la procedencia del código y de los resultados; no son versiones adicionales del manuscrito.

## Alcance y comprobaciones

Este archivo contiene datos públicos procesados, código científico y resultados guardados.
La sección `benchmark/` procede del paquete editorial v9 y conserva la evaluación
comparativa, la aplicación de los pronósticos al cáncer de mama, los escenarios de
desplazamiento de mortalidad y LoRA. La sección `c50/` contiene los análisis exploratorios
de tendencia y lugar de defunción del paquete corregido v11, más APVP y proporciones por localidad de v12.

En esta preparación se verificaron las firmas SHA-256 de los doce archivos NPZ frente
al manifiesto original. Se abrieron todos con `allow_pickle=False`: cada uno contiene
nueve modelos, 2 870 series, tres cuantiles y 36, 48 o 60 meses según el origen.
Todos los valores de estas matrices son finitos. El ensamble se calcula a partir de
los pronósticos individuales, no ocupa una décima matriz en los NPZ.

También se cotejaron las copias con sus fuentes, se revisaron rutas y sintaxis de los
ejecutores y se verificó el archivo ZIP. **No se reejecutó la comparación predictiva ni se reentrenaron modelos.** Los APVP y las proporciones por localidad sí se recalcularon desde los microdatos para esta integración. Los CSV y JSON son resultados
de ejecuciones anteriores; su inclusión y su firma no equivalen a una nueva validación
científica. La reproducción C50 previa fue documentada con 10 000 simulaciones,
semilla 20260930 y dos hilos. La integridad del archivo actual puede comprobarse con:

```sh
python3 verificar_integridad.py
```

## Recálculo de la evaluación y los pronósticos

Instalar `benchmark/requirements.txt` en un entorno separado (entorno fuente:
Python 3.11.16). Desde cualquier directorio:

```sh
python -m pip install -r benchmark/requirements.txt
python benchmark/recalcular.py --listar
python benchmark/recalcular.py
```

Usar una copia de trabajo: el recálculo sobrescribe resultados derivados en esa copia.
`pronosticos/publicados/` contiene los seis archivos que sustentan las cifras principales.
`pronosticos/nueva_ejecucion_semillas/` contiene seis archivos de la ejecución con semillas
fijas utilizada para la comparación de reproducibilidad. Su manifiesto SHA-256 original
se conserva en `benchmark/manifiesto_pronosticos_sha256.txt`.

El ejecutor recalcula métricas, intervalos, detección, escenarios, sensibilidad de
Friedman, agregados de LoRA, derechohabiencia y la comparación entre ejecuciones.
Genera cifras en `resultados/numeros_v9.json` y tablas CSV en `documentos/tablas/`.
Ese último directorio contiene únicamente tablas de datos, no manuscritos.
`codigo/documentos/referencias.py` se conserva porque la tabla de parámetros importa
su lista bibliográfica; no es un generador de documentos.

`entradas/` contiene productos guardados de la canalización de cáncer de mama, LoRA,
el clasificador de causa y costos de cómputo. El recálculo los reutiliza; no los vuelve
a generar desde entrenamiento. `codigo/modelos/`, `codigo/lora/` y
`codigo/construccion_datos/` se conservan como fuentes de estos métodos, fuera del
ejecutor. El entrenamiento requiere sus propias rutas de trabajo, pesos y recursos;
véase `benchmark/versiones_entrenamiento.txt`. No se incluye un entorno de entrenamiento
ni se promete reproducción desde todos los microdatos nacionales sin preparación adicional.

## Recálculo de los análisis exploratorios C50

Usar otro entorno para `c50/requirements.txt` (entorno de reproducción previa:
Python 3.12.14). Las versiones de ambos entornos se conservan separadas.

```sh
python -m pip install -r c50/requirements.txt
python c50/recalcular.py --listar
python c50/recalcular.py
```

El ejecutor usa catálogos incluidos, reconstruye la base analítica, reproduce la ITS
previa en una copia aislada, ejecuta tendencia y lugar de defunción, y produce tablas,
`resultados/numeros_v11.json` y verificaciones. No necesita descargar catálogos ni
una versión anterior completa del manuscrito. La ruta `paquete_original/` conserva
solo las entradas científicas requeridas, con sus nombres originales. El plan de la
ITS previa se incluye porque su firma es una entrada de la comprobación de procedencia.
En los metadatos guardados de esa ITS se sustituyó una ruta personal absoluta por
una ruta relativa; todas sus cifras se conservaron.

Para regenerar figuras científicas, instalar `c50/requirements_figuras.txt` y añadir
`--figuras`. Se usa Arial si está disponible; de otro modo DejaVu Sans. El nombre
interno `Figura6_C50_tendencia_y_lugar` corresponde a la **figura 5** del manuscrito
actual; la adaptación LoRA corresponde a la figura 6. Los números internos de archivos
no deben interpretarse como el orden definitivo de las figuras del artículo.

Las tasas por lugar de 2024 son observadas, sin corrección global. La ponderación
por completitud en sensibilidades de probabilidades de lugar y la corrección global
de la ITS nacional son procedimientos distintos. El análisis separado del código 99
conserva diagnóstico de separación y descriptivos, sin OR, intervalos ni valores p.
Los planes y el registro de decisiones en `c50/plan/` documentan el carácter exploratorio.

## Datos y contenido excluido

Las fuentes son los datos abiertos de defunciones del INEGI y los denominadores
World Population Prospects. Los catálogos y URL de procedencia están en
`c50/fuentes_inegi/` y `benchmark/datos/catalogos_inegi/`. La tabla de microdatos C50
es el extracto público procesado que ya sustentaba los análisis: no contiene nombres,
CURP, domicilios ni identificadores directos añadidos. Se conservan sus claves
geográficas y variables analíticas públicas. No se eliminan filas idénticas de forma
automática. La base analítica en pickle no se distribuye: se vuelve a construir
con `validacion_datos.py`.

No se incluyen manuscritos históricos, fuentes Word, portadas, cartas, firmas,
prompts, entornos virtuales ni cachés. Este ZIP es un apoyo científico separado de
los documentos listos para envío. Su manifiesto permite verificar todos los archivos
distribuidos; el informe de preparación y procedencia permanece fuera de la entrega.

## Ampliación v12 de APVP y localidad

Para repetir solo los indicadores añadidos, sin ajustar modelos anteriores:

```sh
python c50/codigo/c50_apvp_localidad_v12.py
```

El script escribe `c50/resultados/v12_apvp_localidad/`: APVP anuales y mensuales, tres variantes de tendencia y proporciones por localidad/año con sensibilidad del sitio99. No usa conexión a internet ni archivos fuera de este paquete. Las fuentes del límite75, los pesosOMS y las definiciones se documentan en métodosS12 y tablaS25 del manuscrito. La procedencia, exclusiones del paquete de análisis adicional y carácter retrospectivo están en `c50/plan/INTEGRACION_V12_2026-10-01.txt`. El archivo `validacion.json` documenta la comparación con las auditorías previas; no se regenera al repetir únicamente el cálculo.


Actualización editorial v13: se aclararon entrenamiento local, adaptación LoRA y nombres de las variantes. Se mantienen los resultados analíticos y los doce archivos NPZ. editorial_v13/ conserva fuentes anónimas v12, textos, figuras 6 y S3 y scripts para regenerar esta edición. Véase editorial_v13/LEEME_EDITORIAL.txt.
