"""Tabla de correspondencia código-etiqueta de sitio de ocurrencia y de claves de entidad no estándar, por archivo anual.
Agrupación analítica validada contra las etiquetas oficiales."""
import glob, re, json, pandas as pd, numpy as np
C = '../fuentes_inegi/catalogos/'
GRUPO = {**{k: 'Unidad pública' for k in [1, 2, 3, 4, 5, 6, 7, 8, 13]}, 9: 'Unidad privada', 11: 'Hogar',
         10: 'Otro/NE', 12: 'Otro/NE', 99: 'Otro/NE'}
rows = []
for y in range(2012, 2025):
    f = glob.glob(f'{C}{y}_desitiodefun.csv') + glob.glob(f'{C}{y}_sitio_ocurrencia.csv')
    t = pd.read_csv(f[0], encoding='utf-8-sig')
    for _, r in t.iterrows():
        rows.append({'file_year': y, 'archivo': f[0].split('/')[-1][5:], 'codigo': int(r.CVE), 'etiqueta_oficial': r.DESCRIP.strip(),
                     'grupo_analitico': GRUPO.get(int(r.CVE), 'SIN GRUPO')})
s = pd.DataFrame(rows); s.to_csv('../tablas/tablaS_catalogo_sitio_ocurrencia.csv', index=False)
assert (s.grupo_analitico != 'SIN GRUPO').all()
wide = s.pivot(index='codigo', columns='file_year', values='etiqueta_oficial')
cambios = {int(c): sorted(set(wide.loc[c].dropna())) for c in wide.index if wide.loc[c].dropna().nunique() > 1 or wide.loc[c].isna().any()}
# entidades: claves de entidad fuera de 1-32
ent = []
for y in range(2012, 2025):
    f = [g for g in glob.glob(f'{C}{y}_*') if 'decateml' in g.lower() or 'entidad_municipio' in g.lower()][0]
    t = pd.read_csv(f, encoding='utf-8-sig', dtype=str, encoding_errors='replace')
    t.columns = [c.strip().lower() for c in t.columns]
    for c in ['cve_ent', 'cve_mun', 'cve_loc']: t[c] = t[c].str.strip()
    e = t[(t.cve_mun.astype(int) == 0) & (t.cve_loc.astype(int) == 0)]
    for _, r in e.iterrows():
        k = int(r.cve_ent)
        if k > 32: ent.append({'file_year': y, 'clave_entidad': k, 'descripcion': str(r.nom_loc).strip()})
E = pd.DataFrame(ent); E.to_csv('../tablas/tablaS_catalogo_entidades_no_estandar.csv', index=False)
json.dump({'cambios_etiqueta_sitio': cambios, 'n_filas_entidades_no_estandar': len(ent)}, open('../resultados/catalogos_resumen.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(cambios, ensure_ascii=False, indent=0)); print(E.pivot(index='clave_entidad', columns='file_year', values='descripcion').T.to_string())
