# Guía breve para investigadores médicos

**¿Qué estudia el artículo?** Cuántas defunciones se esperaban en México sin pandemia, estimadas con métodos
estadísticos clásicos y con inteligencia artificial, y cómo se compara esa expectativa con lo observado en el
cáncer de mama en mujeres de 20 años o más. Incluye análisis exploratorios de tendencia, lugar de defunción, años
potenciales de vida perdidos antes de los 75 años y tamaño de localidad.

**¿Qué permite este repositorio?**

1. Confirmar que los archivos son los mismos que sustentan el artículo (nivel 1).
2. Volver a dibujar las 12 figuras a partir de los resultados guardados (nivel 2).
3. Recalcular las tablas y cifras desde los pronósticos guardados y desde el extracto público de defunciones por
   cáncer de mama del INEGI (nivel 3).

**¿Qué no hace?** No vuelve a entrenar los modelos de IA (nivel 4): eso requiere descargar modelos de terceros,
otro entorno y más tiempo. El código está incluido y documentado, pero esa parte no se comprobó.

**Cómo leer los resultados.** Las diferencias entre lo observado y lo esperado en cáncer de mama tienen intervalos
de predicción amplios y fueron compatibles con cero. Los análisis de tendencia y lugar de defunción son
exploratorios y ecológicos: describen poblaciones, no causas ni la calidad de la atención individual. Las
proporciones de defunciones en el hogar por tamaño de localidad son proporciones entre fallecidas, no tasas.

**¿Qué necesito para repetirlo?** Una computadora con Python, los dos paquetes de datos y unos minutos (el análisis
más largo tarda unos 4 minutos en una computadora portátil reciente). Los pasos están en el `README.md`.
