"""Estratos de volumen (defunciones/mes) para la calibración conformal Mondrian (v7).
Para cualquier análisis cuya prueba cae en 2015-2019 los estratos se fijan con 2012-2014 (años anteriores a todas las pruebas);
la aplicación 2020-2024 (origen dic-2019) usa 2012-2019. Columnas del corpus: '0'..'155' = ene-2012..dic-2024."""
import numpy as np, pandas as pd
BINS = [0, 10, 25, 100, 300, 1000, 1e7]; LAB = ['0-10', '10-25', '25-100', '100-300', '300-1000', '>1000']
def estrato(m): return pd.cut(np.atleast_1d(m), BINS, labels=LAB).astype(str)
def vol_corpus(C, base='2012-2014'):
    Y = C[[str(i) for i in range(156)]].values.astype(float)
    n = {'2012-2014': 36, '2012-2019': 96}[base]
    return pd.Series(estrato(Y[:, :n].mean(1)), index=C.index, name='vol')
