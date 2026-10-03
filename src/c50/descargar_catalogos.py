"""Descarga, de los ZIP oficiales de datos abiertos del INEGI (defunciones registradas 2012-2024), los catálogos de
sitio de ocurrencia de la defunción y de entidad/municipio/localidad. Guarda copia íntegra, URL, miembro, fecha de
consulta y SHA-256. Uso: python descargar_catalogos.py (requiere red)."""
import json, hashlib, datetime, os, zip_remoto
m = json.load(open('../fuentes_inegi/miembros_zip.json'))
os.makedirs('../fuentes_inegi/catalogos', exist_ok=True)
reg = {}
for y, v in m.items():
    z = zip_remoto.abrir(v['url']); reg[y] = {'url': v['url'], 'archivos': []}
    for n in v['members']:
        b = n.split('/')[-1].lower()
        if b in ('desitiodefun.csv', 'sitio_ocurrencia.csv') or b.startswith('decateml') or b.startswith('entidad_municipio_localidad'):
            data = z.read(n); dst = f'../fuentes_inegi/catalogos/{y}_{n.split("/")[-1]}'
            open(dst, 'wb').write(data)
            reg[y]['archivos'].append({'miembro': n, 'copia': os.path.basename(dst), 'bytes': len(data),
                                       'sha256': hashlib.sha256(data).hexdigest(),
                                       'consulta': datetime.datetime.now().astimezone().isoformat(timespec='seconds')})
    print(y, [a['copia'] for a in reg[y]['archivos']])
json.dump(reg, open('../fuentes_inegi/fuentes_catalogos.json', 'w'), indent=1, ensure_ascii=False)
