"""Lista los miembros de catálogos de los ZIP oficiales de defunciones registradas del INEGI 2012-2024 (requiere red).
Las URL provienen de catalogos_derechohab_fuentes.json del paquete v9 y se guardan en ../fuentes_inegi/miembros_zip.json."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zip_remoto
URLS = json.load(open('../fuentes_inegi/urls_datos_abiertos_inegi.json'))
out = {}
for y, url in URLS.items():
    z = zip_remoto.abrir(url); out[y] = {'url': url, 'members': [n for n in z.namelist() if 'catal' in n.lower() or 'diccion' in n.lower()]}
json.dump(out, open('../fuentes_inegi/miembros_zip.json', 'w'), indent=1)
