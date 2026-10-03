"""P5 (v6; v7: documenta los códigos fuera del catálogo con año de ocurrencia, de registro y archivo fuente). Derechohabiencia de las defunciones por cáncer de mama (mujeres ≥ 20 años) por código y año, contrastada con los
catálogos oficiales del INEGI de cada año (datos/catalogos_inegi/, extraídos de los ZIP de datos abiertos EDR; fuentes en
catalogos_derechohab_fuentes.json; extracción con codigo/construccion_datos/zip_remoto.py).
Agrupación usada en el análisis: con seguridad social 2, 3, 4, 5, 6, 10; sin seguridad social 1, 7, 9, 12; otra o no especificada 8, 11, 99, vacío
(redistribuida proporcionalmente cada mes). Salidas: resultados/p5_derechohab_*.csv"""
import numpy as np, pandas as pd, glob, re
d = pd.read_csv('datos/c50_raw_2012_2024.csv.gz', dtype=str)
for c in ['sexo', 'edad', 'anio_ocur', 'mes_ocurr']: d[c] = pd.to_numeric(d[c], errors='coerce')
d['age'] = np.where((d.edad >= 4000) & (d.edad < 4998), d.edad - 4000, np.nan)
w = d[(d.sexo == 2) & (d.age >= 20) & d.anio_ocur.between(2012, 2024) & d.mes_ocurr.between(1, 12)].copy()
w['dh'] = w.derechohab.fillna('vacío').str.strip().replace('', 'vacío')
T = pd.crosstab(w.dh, w.anio_ocur.astype(int)); T.index.name = 'codigo'; T.to_csv('resultados/p5_derechohab_codigos_por_anio.csv')
grp = lambda c: 'Con seguridad social' if c in {'2', '3', '4', '5', '6', '10'} else ('Sin seguridad social' if c in {'1', '7', '9', '12'} else 'Otra o no especificada')
w['g'] = w.dh.map(grp)
H = pd.crosstab(w.g, w.anio_ocur.astype(int), normalize='columns').mul(100).round(1); H.index.name = 'grupo'; H.to_csv('resultados/p5_derechohab_grupos_por_anio_pct.csv')
# catálogos oficiales por año (año de registro del archivo)
cat = {}
for f in sorted(glob.glob('datos/catalogos_inegi/derechohab_*.csv')):
    y = int(re.search(r'derechohab_(\d{4})_', f).group(1))
    c = pd.read_csv(f, encoding='latin1', dtype=str); c.columns = [x.replace('ï»¿', '').strip() for x in c.columns]
    cat[y] = {k.strip(): v.strip() for k, v in zip(c.CVE, c.DESCRIP)}
codes = sorted({k for v in cat.values() for k in v} | set(T.index), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
C = pd.DataFrame({y: [cat[y].get(k, '—') for k in codes] for y in sorted(cat)}, index=codes); C.index.name = 'codigo'
C.to_csv('resultados/p5_catalogos_derechohab_por_anio.csv')
# códigos observados ausentes del catálogo del archivo (año de registro) en que aparecen; se tabulan por año de ocurrencia,
# de registro y archivo fuente. El catálogo aplicable es el del archivo (año de registro); v7 verifica además el diccionario de datos.
w['fy'] = pd.to_numeric(w.file_year, errors='coerce').astype('Int64'); w['ar'] = pd.to_numeric(w.anio_regis, errors='coerce').astype('Int64')
rows = []
for (k, fy), g in w.groupby(['dh', 'fy']):
    if k != 'vacío' and cat.get(int(fy), {}).get(k) is None:
        for (ao, ar), gg in g.groupby(['anio_ocur', 'ar']):
            n_tot = int((w.anio_ocur == ao).sum()); ss = int(w[(w.anio_ocur == ao)].g.eq('Con seguridad social').sum())
            rows.append(dict(codigo=k, anio=int(ao), anio_registro=int(ar), archivo=int(fy), defunciones=len(gg),
                             pct_defunciones_anio=len(gg) / n_tot * 100, tratamiento='con seguridad social (ISSFAM en el catálogo 2023)',
                             pct_con_ss_anio=ss / n_tot * 100, pct_con_ss_si_no_especificado=(ss - len(gg)) / n_tot * 100))
pd.DataFrame(rows).to_csv('resultados/p5_codigos_sin_catalogo.csv', index=False)
obs_no_cat = rows
print(C.to_string()); print(T.to_string()); print(H.to_string()); print(obs_no_cat)
