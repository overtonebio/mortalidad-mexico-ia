"""Coordinación y presentación del notebook; los análisis viven en src/.

No descarga datos ni modelos. Las tablas siempre leen la ejecución elegida:
una entrada guardada se declara como tal, nunca sustituye a un cálculo fallido.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

MIEMBROS = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']
ESTADOS = ('recalculado', 'regenerado desde resultados guardados', 'documental')


def raiz_repositorio(inicio=None):
    p = Path(inicio or Path.cwd()).resolve()
    for candidato in (p, *p.parents):
        if (candidato / 'scripts/gmm.py').is_file() and (candidato / 'configs/pasos.json').is_file():
            return candidato
    raise FileNotFoundError('Abra el notebook desde este repositorio (raíz o notebooks/).')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cargar_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def guardar_json(path, contenido):
    Path(path).write_text(json.dumps(contenido, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def enlace(path, etiqueta=None):
    # Enlaces relativos al notebook, válidos también en JupyterLab.
    from urllib.parse import quote
    rel = os.path.relpath(path, Path(__file__).parent)
    return f'<a href="{quote(rel, safe="/#")}">{escape(etiqueta or rel)}</a>'


def comprobar_entornos(repo, python_benchmark, python_c50):
    """Comprueba ejecutables y versiones reales; no confía en el nombre del venv."""
    filas, problemas, inventarios = [], [], {}
    for nombre, exe, version, requisitos in [
        ('benchmark', python_benchmark, (3, 11), 'benchmark-recalculo.txt'),
        ('C50 / notebook', python_c50, (3, 12), 'c50-figuras.txt'),
    ]:
        exe = Path(exe).expanduser().absolute()  # preservar el enlace del venv
        if not exe.is_file():
            problemas.append(f'{nombre}: falta {exe}. Prepare el entorno indicado en README.')
            continue
        programa = ('import sys,json,importlib.metadata as m; '
                    'print(json.dumps({"python":list(sys.version_info[:3]),"paquetes":'
                    '{d.metadata["Name"]:d.version for d in m.distributions()}}))')
        r = subprocess.run([str(exe), '-c', programa], capture_output=True, text=True, check=True)
        info = json.loads(r.stdout)
        inventarios[nombre] = info
        filas.append({'Entorno': nombre, 'Paquete': 'Python', 'Versión': '.'.join(map(str, info['python']))})
        if tuple(info['python'][:2]) != version:
            problemas.append(f'{nombre}: Python {version[0]}.{version[1]} requerido; encontrado {info["python"]}.')
        esperados = {}
        def leer(req):
            for linea in (repo / 'environments' / req).read_text().splitlines():
                linea = linea.strip()
                if linea.startswith('-r '):
                    leer(linea[3:].strip())
                elif '==' in linea and not linea.startswith('#'):
                    k, v = linea.split('==', 1)
                    esperados[k.lower()] = v
        leer(requisitos)
        versiones = {k.lower().replace('_', '-'): v for k, v in info['paquetes'].items()}
        for paquete, esperado in esperados.items():
            encontrado = versiones.get(paquete)
            filas.append({'Entorno': nombre, 'Paquete': paquete, 'Versión': encontrado or 'ausente'})
            if encontrado != esperado:
                problemas.append(f'{nombre}: {paquete}=={esperado} requerido; encontrado {encontrado}.')
        if nombre.startswith('C50'):
            for paquete in ['nbformat', 'nbclient', 'ipykernel', 'pypdfium2']:
                if paquete not in versiones:
                    problemas.append(f'Instale environments/notebook.txt: falta {paquete}.')
    if problemas:
        raise RuntimeError('\n'.join(problemas))
    return pd.DataFrame(filas), inventarios


def nueva_ejecucion(repo, datos, base):
    datos, base = Path(datos).resolve(), Path(base).resolve()
    for protegida in [datos, repo / 'results', repo / 'src', repo / 'scripts', repo / 'notebooks', repo / 'configs']:
        if base.is_relative_to(protegida) or protegida.is_relative_to(base):
            raise ValueError(f'La salida invade una ruta protegida: {base}')
    if not datos.is_dir():
        raise FileNotFoundError(f'Faltan los datos: {datos}. Siga docs/DESCARGA_Y_EJECUCION.md.')
    nombre = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = base / nombre
    out.mkdir(parents=True, exist_ok=False)
    for sub in ['logs_notebook', 'tablas_notebook', 'imagenes_notebook']:
        (out / sub).mkdir()
    guardar_json(out / 'registro_notebook.json', {'inicio_utc': datetime.now(timezone.utc).isoformat(), 'pasos': []})
    return out


def ejecutar(repo, salida, nombre, comando, cwd=None, codigos=(0,)):
    """Ejecuta sin shell, conserva el log íntegro y detiene errores de análisis."""
    log = salida / 'logs_notebook' / f'{nombre}.log'
    env = os.environ.copy()
    env.pop('PYTHONSAFEPATH', None)
    env.pop('ESTRATOS', None)
    env.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    print(f'{nombre}: en ejecución…', flush=True)
    inicio = time.monotonic()
    with log.open('w', encoding='utf-8') as f:
        r = subprocess.run(list(map(str, comando)), cwd=cwd or repo, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    segundos = round(time.monotonic() - inicio, 2)
    registro = cargar_json(salida / 'registro_notebook.json')
    registro['pasos'].append({'nombre': nombre, 'comando': list(map(str, comando)),
                             'codigo_salida': r.returncode, 'segundos': segundos,
                             'log': str(log.relative_to(salida))})
    guardar_json(salida / 'registro_notebook.json', registro)
    cola = log.read_text(errors='replace').splitlines()
    if r.returncode not in codigos:
        raise RuntimeError(f'{nombre} falló ({r.returncode}). Log: {log}\n' + '\n'.join(cola[-20:]))
    print(f'{nombre}: código {r.returncode}; {segundos:g} s. Registro completo en logs_notebook/.', flush=True)
    # Las comparaciones e integridad merecen salida visible, sin volcar todos los scripts.
    if nombre.startswith(('verificar', 'comparar', 'reconstruir')):
        print('\n'.join(cola[-8:]).replace(str(repo), '.'))
    return r.returncode


def instantanea_entradas(repo, datos):
    entradas = cargar_json(repo / 'configs/datos_esperados.json')['archivos']
    referencias = cargar_json(repo / 'results/reference/SHA256SUMS.json')
    return {'datos': {r: sha256(datos / r) for r in entradas},
            'referencias': {r: sha256(repo / 'results/reference' / r) for r in referencias}}


def artistas_numericos(fig):
    """Datos y geometría en coordenadas de datos, independientes de la tipografía."""
    def array(x):
        return np.asarray(x, dtype=float).tolist()
    return [{'xlim': list(a.get_xlim()), 'ylim': list(a.get_ylim()),
             'lines': [(array(z.get_xdata(orig=False)), array(z.get_ydata(orig=False))) for z in a.lines],
             'patches': [(array(z.get_path().vertices), array(z.get_patch_transform().get_matrix())) for z in a.patches],
             'collections': [(array(z.get_offsets()), [array(v.vertices) for v in z.get_paths()]) for z in a.collections]}
            for a in fig.axes]


def _rotulos_portables(repo, trabajo, codigo, destino):
    """Exporta S1/S2 y C50 conservando artistas; no exige hashes de fuentes macOS.

    El control PNG histórico se registra separadamente como comparación tipográfica,
    no como prueba de equivalencia de los datos.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    sys.path.insert(0, str(codigo))
    from estilo_figuras import apply_figure_style
    apply_figure_style()
    os.chdir(trabajo / 'benchmark')
    def ns(path):
        g = {'__file__': str(path), '__name__': '__notebook_generador__'}
        exec(compile(path.read_text(), str(path), 'exec'), g)
        return g
    def captura(path, funcion):
        g = ns(path)
        figuras = []
        g['_save'] = lambda f, out: figuras.append(f)
        g[funcion](out='captura')
        return figuras[0]
    informe = {}
    cajas = cargar_json(repo / 'configs/cajas_S1_S2_v20.json')
    for n, funcion in [('S1', 'figS1b'), ('S2', 'figS2b')]:
        old = captura(codigo / 'figuras_base_v9.py', funcion)
        control = captura(codigo / 'figuras_base_v9_DP_original.py', funcion)
        new = control if n == 'S1' else captura(codigo / 'figuras_base_v9_DP_rotulos_20261017.py', funcion)
        assert artistas_numericos(old) == artistas_numericos(control) == artistas_numericos(new), n
        for f in [old, control, new]:
            f.canvas.draw()
        box = old.get_tightbbox(old.canvas.get_renderer()).padded(.05)
        w, h = cajas[n]['output_inches']
        box = Bbox.from_bounds(box.x0 + (box.width-w)/2, box.y0 + (box.height-h)/2, w, h)
        informe[n] = {'artistas_numericos_identicos': True}
        if n == 'S2':
            png = destino / 'control_S2_v19.png'
            control.savefig(png, dpi=300, bbox_inches=box, pad_inches=0)
            informe[n]['control_tipografico_png_coincide'] = sha256(png) == cajas[n]['control_png_sha256']
            t = new.get_tightbbox(new.canvas.get_renderer())
            box = Bbox([[box.x0, box.y0], [box.x1, max(box.y1, t.y1+.05)]])
        new.savefig(destino / f'Figura_{n}.pdf', dpi=300, bbox_inches=box, pad_inches=0)
        plt.close('all')
    # Comprobar también los artistas de los cuatro gráficos C50 antes/después.
    os.chdir(trabajo / 'c50/codigo')
    actualizado = Path.cwd() / 'figuras_c50_notebook_rotulos.py'
    shutil.copy2(codigo / 'figuras_c50_rotulos_20261017.py', actualizado)
    original_savefig, original_close = plt.Figure.savefig, plt.close
    capturas = {}
    def guardar(fig, filename, *args, **kwargs):
        if str(filename).endswith('.pdf'):
            capturas[Path(filename).stem] = artistas_numericos(fig)
    try:
        plt.Figure.savefig = guardar
        ns(Path.cwd() / 'figuras_c50.py')
        anteriores = dict(capturas)
        capturas.clear()
        ns(actualizado)
        assert anteriores == capturas, 'Los rótulos C50 cambiaron artistas numéricos'
        informe['C50'] = {'artistas_numericos_identicos': True, 'figuras': sorted(capturas)}
    finally:
        plt.Figure.savefig, plt.close = original_savefig, original_close
        plt.close('all')
    # Ahora guardar las versiones editoriales comprobadas.
    ns(actualizado)
    guardar_json(destino / 'equivalencia_rotulos.json', informe)


def actualizar_rotulos(repo, salida, python_c50):
    fig = salida / 'figuras'
    codigo = fig / 'codigo_rotulos'
    shutil.copytree(repo / 'src/figuras', codigo)
    for p in (repo / 'src/v20/figuras').glob('*.py'):
        shutil.copy2(p, codigo / p.name)
    trabajo = fig / 'trabajo'
    shutil.copytree(fig / 'figuras', fig / 'figuras_base_historica')
    destino = fig / 'rotulos_documentos'
    destino.mkdir()
    ejecutar(repo, salida, 'rotulos_F3', [python_c50, codigo / 'ejecutar_F3_rotulos_20261017.py', codigo, destino / 'F3'], cwd=trabajo / 'benchmark')
    ejecutar(repo, salida, 'rotulos_F6_S3', [python_c50, codigo / 'actualizar_figuras_rotulos_20261017.py', '--benchmark', trabajo / 'benchmark', '--out', destino / 'F6_S3'])
    ejecutar(repo, salida, 'rotulos_S1_S2_C50', [python_c50, Path(__file__), '_rotulos', repo, trabajo, codigo, destino])
    archivos = {'Figura_3': destino / 'F3/Figura_3.pdf', 'Figura_6': destino / 'F6_S3/Figura_6.pdf',
                'Figura_S3': destino / 'F6_S3/Figura_S3.pdf',
                'Figura_S1': destino / 'Figura_S1.pdf', 'Figura_S2': destino / 'Figura_S2.pdf'}
    for n, origen in [('5', 'Figura6_C50_tendencia_y_lugar'), ('S4', 'FiguraS_C50_variantes_tendencia'),
                      ('S5', 'FiguraS_C50_lugar_anual_y_sensibilidades'), ('S6', 'FiguraS_C50_residuos')]:
        archivos[f'Figura_{n}'] = trabajo / 'c50/figuras' / f'{origen}.pdf'
    for n, origen in archivos.items():
        shutil.copy2(origen, fig / 'figuras' / f'{n}.pdf')


def metadatos_documentos(repo):
    return cargar_json(repo / 'notebooks/documentos_fuente.json')


def mostrar_figura(repo, salida, numero):
    import pypdfium2 as pdfium
    meta = metadatos_documentos(repo)['figuras'][numero]
    pdf = salida / 'figuras/figuras' / f'Figura_{numero}.pdf'
    png = salida / 'imagenes_notebook' / f'Figura_{numero}.png'
    with pdfium.PdfDocument(pdf) as doc:
        page = doc[0]
        bitmap = page.render(scale=180/72)
        im = bitmap.to_pil()
        im.save(png)
        ancho = min(1100, im.width)
        im.close()
        bitmap.close()
        page.close()
    estado = 'documental' if numero == '1' else ('regenerado desde resultados guardados' if numero in ['6', 'S2', 'S3'] else 'recalculado')
    b, c = salida / 'benchmark/benchmark', salida / 'c50/c50'
    def entrada(nombre):
        return salida / 'figuras/trabajo/benchmark/entradas' / nombre
    fuentes = {
        '1': [repo / 'src/figuras/figura_diseno_v11.py'],
        '2': [b / 'resultados/benchmark/numeros_repro.json', b / 'resultados/benchmark/friedman_por_serie.json'],
        '3': [b / 'resultados/p2_deteccion_resultados.csv', b / 'resultados/benchmark/atlas_repro.csv.gz'],
        '4': [b / 'resultados/p1_intervalos_mama_entre_series.csv', entrada('tsfm_contrafactual_mensual.csv')],
        '5': [c / 'resultados/lugar_defuncion.json', c / 'resultados/tendencia_ampliaciones.json'],
        '6': [entrada('finetune2_resultados.csv.gz'), entrada('finetune_estricto_resultados.csv.gz')],
        'S1': [b / 'resultados/p1_intervalos_mama_entre_series.csv', b / 'resultados/p3_cosecha.json', entrada('lugar_defuncion_C50.csv')],
        'S2': [entrada('subregistro_resultados.csv'), entrada('subregistro_calibracion.csv'), entrada('subregistro_shap.csv')],
        'S3': [entrada('finetune_resultados.csv.gz'), entrada('finetune_rel_por_volumen.csv')],
        'S4': [c / 'tablas/tablaS_tendencia_ampliaciones.csv', c / 'reproduccion/pk/exploracion_previa/estimaciones_ITS.csv'],
        'S5': [c / 'tablas/tablaS_lugar_proporcion_ajustada_anual.csv', c / 'tablas/tablaS_lugar_proporciones_estandarizadas.csv'],
        'S6': [c / 'resultados/residuos_ITS_principal.csv', c / 'resultados/tendencia_ampliaciones.json'],
    }[numero]
    codigo = re.search(r'src/[\w/]+\.py', meta.get('codigo', ''))
    link_codigo = enlace(repo / codigo.group(), 'Código del generador') if codigo else escape(meta.get('codigo', ''))
    display(HTML(f'<h3 id="figura-{numero}">Figura {numero}</h3><p><b>{estado}</b> · '
                 + enlace(pdf, 'PDF vectorial') + ' · ' + link_codigo + '</p>'))
    display(Image(data=png.read_bytes(), width=ancho, alt=f'Figura {numero}: {meta["titulo"]}'))
    display(Markdown(meta['pie']))
    display(HTML('<p><small>' + escape(meta.get('alcance', '')) + '</small></p><details><summary>Datos de esta figura</summary><p>'
                 + ' · '.join(enlace(p) for p in fuentes) + '</p></details>'))
    return {'elemento': f'Figura {numero}', 'estado': estado, 'archivo': str(pdf.relative_to(salida)),
            'sha256': sha256(pdf), 'fuentes': [str(p) for p in fuentes], 'codigo': meta.get('codigo', ''), 'nota': meta.get('alcance', '')}


def _ic(v, dec=1):
    def fmt(x):
        # Evitar que un límite pequeño no nulo se convierta en cero al mostrarlo.
        d = max(dec, int(-np.floor(np.log10(abs(x))))) if 0 < abs(x) < .5*10**(-dec) else dec
        return f'{x:.{d}f}'
    return f'{fmt(v[0])} ({fmt(v[1])} a {fmt(v[2])})'


def tabla_especial(numero, repo, salida, datos):
    """Adaptaciones editoriales de tablas procedentes de varios CSV o JSON."""
    c = salida / 'c50/c50'
    def csv(rel):
        return pd.read_csv(c / rel)
    if numero in ['S13', 'S16']:
        d = metadatos_documentos(repo)['tablas_documentales'][numero]
        return pd.DataFrame(d['filas'], columns=d['columnas'])
    if numero == 'S5':
        r = pd.read_csv(datos / 'benchmark/entradas/subregistro_desempeno.csv').iloc[0]
        return pd.DataFrame([
            ['Defunciones de entrenamiento (2012-2018)', f'{r.n_train:.0f}'],
            ['Defunciones de validación (2019)', f'{r.n_valid:.0f}'],
            ['Proporción de C50 en validación (%)', f'{100*r.prev_valid:.2f}'],
            ['Iteraciones (parada temprana)', f'{r.best_iter:.0f}'],
            ['AUC', f'{r.AUC:.3f}'], ['AUC, modelo solo con edad', f'{r.AUC_edad:.3f}'],
            ['AUPRC', f'{r.AUPRC:.3f}'], ['AUPRC, modelo solo con edad', f'{r.AUPRC_edad:.3f}'],
            ['Puntuación de Brier', f'{r.Brier:.3f}'],
            ['Razón de casos predichos/observados', f'{r.calib_ratio:.3f}'],
        ], columns=['Métrica', 'Valor'])
    if numero == 'S17':
        cat = csv('tablas/tablaS_catalogo_sitio_ocurrencia.csv')
        cat['periodo'] = pd.cut(cat.file_year, [2011, 2017, 2022, 2024], labels=['2012-2017', '2018-2022', '2023-2024'])
        t = cat.pivot_table(index=['codigo', 'grupo_analitico'], columns='periodo', values='etiqueta_oficial', aggfunc='first', observed=True).reset_index()
        # El código 13 está ausente de los primeros catálogos, pero tiene cero registros.
        conteos = csv('tablas/tablaS_sitio_codigo_por_archivo.csv').drop(columns='file_year').sum()
        t['Registros C50 2012-2024'] = t.codigo.map(lambda v: int(conteos.get(str(v), 0)))
        return t.fillna('no existe')
    if numero == 'S18':
        t = csv('tablas/tablaS_rezago_registro.csv').set_index('anio_ocur')
        total = t.sum(axis=1)
        resultado = pd.DataFrame({'Año de ocurrencia': t.index, 'Defunciones': total,
                             'Mismo año (%)': 100*t['0']/total, 'Año siguiente (%)': 100*t['1']/total,
                             '≥ 2 años (%)': 100*t.drop(columns=['0', '1']).sum(axis=1)/total}).reset_index(drop=True).round(2)
        resultado['≥ 2 años (%)'] = resultado['≥ 2 años (%)'].astype(object)
        resultado.loc[resultado['Año de ocurrencia'] >= 2023, '≥ 2 años (%)'] = 'no observable'
        return resultado
    if numero == 'S19':
        t = csv('tablas/tablaS_tendencia_ampliaciones.csv')
        previa = pd.read_csv(c / 'reproduccion/pk/exploracion_previa/estimaciones_ITS.csv')
        # Las variantes previas se recalculan en reproducir_its_previa.py.
        t = t[t.analisis != 'principal (control)']
        t = pd.concat([previa, t], ignore_index=True)
        t['Cambio % (IC 95 %)'] = [_ic([r.estimacion_pct, r.ic95_inf, r.ic95_sup], 2) for r in t.itertuples()]
        return t.pivot_table(index=['analisis', 'serie'], columns='parametro', values='Cambio % (IC 95 %)', aggfunc='first').reset_index().rename(columns={
            'analisis': 'Análisis', 'serie': 'Serie', 'pendiente_pre': 'Pendiente previa, %/año',
            'pendiente_post': 'Pendiente posterior, %/año', 'cambio_pendiente': 'Cambio de pendiente, %', 'salto_nivel': 'Cambio de nivel, %'})
    if numero == 'S20':
        t = csv('tablas/tablaS_lugar_razones_momios.csv')
        t = t[(t.analisis == 'principal') | t.analisis.str.startswith('sens_')]
        t = t[t.parametro.isin(['pendiente_pre_anual', 'cambio_nivel', 'cambio_pendiente_anual'])].copy()
        t['RM (IC 95 %)'] = [_ic([r.OR, r.ic95_inf, r.ic95_sup], 3) for r in t.itertuples()]
        p = t.pivot(index=['desenlace', 'analisis'], columns='parametro', values='RM (IC 95 %)')
        conteos = t.groupby(['desenlace', 'analisis']).first()
        p['Eventos / defunciones'] = conteos.n_evento.map(lambda x: f'{x:.0f}') + ' / ' + conteos.n_defunciones.map(lambda x: f'{x:.0f}')
        p['Wald nivel y pendiente, p'] = conteos.wald_nivel_pendiente_p.map(lambda x: f'{x:.3g}')
        return p.reset_index().rename(columns={'desenlace':'Desenlace','analisis':'Análisis','pendiente_pre_anual':'Pendiente previa, RM/año',
                                               'cambio_nivel':'Cambio de nivel, RM','cambio_pendiente_anual':'Cambio de pendiente, RM/año'})
    if numero == 'S21A':
        t = csv('tablas/tablaS_lugar_proporciones_estandarizadas.csv')
        t = t[t.periodo.isin(['2019', 'mar-dic 2020', '2021', '2022', '2023'])]
        principal = t[t.analisis == 'principal'].copy()
        posterior = t[t.analisis == 'anual_post'].set_index(['desenlace', 'periodo'])
        filas = []
        for r in principal.itertuples():
            s = posterior.loc[(r.desenlace, r.periodo)]
            filas.append({'Desenlace':r.desenlace, 'Periodo':r.periodo,
                          'Ajustada, % (IC 95 %)':_ic([r.ajustado_pct, r.aj_ic_inf, r.aj_ic_sup]),
                          'Continuación previa, % (IC 95 %)':_ic([r.contrafactual_pct, r.cf_ic_inf, r.cf_ic_sup]),
                          'Diferencia principal, pp (IC 95 %)':_ic([r.diferencia_pp, r.dif_ic_inf, r.dif_ic_sup]),
                          'Diferencia indicadores posteriores, pp (IC 95 %)':_ic([s.diferencia_pp, s.dif_ic_inf, s.dif_ic_sup])})
        return pd.DataFrame(filas)
    if numero == 'S21B':
        filas = []
        l = cargar_json(c / 'resultados/lugar_defuncion.json')
        for d in ['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE']:
            fila = {'Desenlace': d}
            for prefijo, etiqueta in [('principal', 'Tendencia segmentada'), ('anual_post', 'Indicadores anuales'), ('anios', 'Indicadores de todos los años')]:
                v = l[f'{prefijo}_{d}']['periodos']['2023 menos 2019 (ajustado)']
                fila[etiqueta + ' (pp, IC 95 %)'] = _ic([v['diferencia_pp'], *v['dif_ic']])
            filas.append(fila)
        return pd.DataFrame(filas)
    if numero == 'S22':
        t = csv('tablas/tablaS_lugar_proporciones_estandarizadas.csv')
        t = t[t.periodo.isin(['mar-dic 2020', '2023']) & (t.desenlace != 'Otro/NE')]
        t = t.assign(**{'Diferencia, pp (IC 95 %)': [_ic([r.diferencia_pp, r.dif_ic_inf, r.dif_ic_sup]) for r in t.itertuples()]})
        return t.pivot(index=['analisis', 'desenlace'], columns='periodo', values='Diferencia, pp (IC 95 %)').reset_index()
    if numero == 'S23':
        t = csv('tablas/tablaS_lugar_proporcion_ajustada_anual.csv')
        t['Proporción ajustada, % (IC 95 %)'] = [_ic([r.ajustado_pct, r.ic95_inf, r.ic95_sup]) for r in t.itertuples()]
        t = t.pivot(index='anio', columns='desenlace', values='Proporción ajustada, % (IC 95 %)')
        tasas = csv('tablas/tablaS_tasas_por_lugar_anuales.csv').set_index('anio').round(2)
        return t.join(tasas.add_prefix('TEE '), how='outer').reset_index().fillna('—')
    if numero == 'S24':
        t = csv('tablas/tablaS_tasas_por_lugar_diagnosticos.csv')
        p = t.pivot(index='lugar', columns='rezago', values='lb_pvalue').add_prefix('Ljung-Box, p (').rename(columns=lambda s: s+')')
        return t.groupby('lugar').autocorrelacion_1.first().to_frame().join(p).reset_index()
    if numero == 'S25A':
        return csv('resultados/v12_apvp_localidad/APVP_anual.csv')[['anio', 'APVP_total', 'APVP_tasa_estandarizada']].round({'APVP_total': 1, 'APVP_tasa_estandarizada': 1})
    if numero == 'S25B':
        d = cargar_json(c / 'resultados/v12_apvp_localidad/resumen.json')['ITS']
        filas = []
        for variante, ajustes in d.items():
            for par in ['pre', 'post', 'nivel', 'cambio']:
                v = ajustes[par]
                filas.append({'Variante': variante, 'Parámetro': {'pre':'Pendiente previa','post':'Pendiente posterior','nivel':'Cambio de nivel','cambio':'Cambio de pendiente'}[par],
                              'Cambio % (IC 95 %)': _ic([v['pct'], v['lo'], v['hi']], 2), 'p': '< 0.001' if v['p'] < .001 else f"{v['p']:.3f}"})
        return pd.DataFrame(filas)
    if numero == 'S25C':
        t = csv('resultados/v12_apvp_localidad/localidad_hogar.csv')
        return pd.DataFrame({'Habitantes': t.localidad, 'Año': t.anio,
                             'Hogar / total': t.hogar.astype(str) + '/' + t.n.astype(str),
                             'Hogar %': t.hogar_pct.round(1), 'Sitio 99 n': t.lugar_no_especificado,
                             'Hogar % sin 99': t.hogar_pct_sin99.round(1)})
    raise KeyError(f'Tabla especial sin adaptador: {numero}')


def catalogo_tablas(repo, salida, datos):
    """Una fila por identificador impreso, con fuente real y límites de recálculo."""
    matriz = pd.read_csv(repo / 'docs/MATRIZ_TRAZABILIDAD.csv').fillna('')
    fuente_directa = {
        '1': 'benchmark/benchmark/documentos/tablas/tabla1_manuscrito.csv',
        '2': 'benchmark/benchmark/documentos/tablas/tabla2_v9.csv',
        'S1': 'reconstruccion/tablaS1_grupos_CIE10.csv',
        'S2A': 'reconstruccion/tablaS2A_validacion48_benchmark.csv',
    }
    guardadas = {'S3':'conformal_variantes_placebo.csv', 'S5':'subregistro_desempeno.csv',
                 'S6':'cqr_recalibracion_2016.csv', 'S7':'tablaS7_costo_computacional.csv',
                 'S8':'tablaS8_conteos_mama_todos_modelos.csv'}
    especiales = {'S5','S13','S16','S17','S18','S19','S20','S21A','S21B','S22','S23','S24','S25A','S25B','S25C'}
    fuentes_especiales = {
        'S17': ['tablas/tablaS_catalogo_sitio_ocurrencia.csv', 'tablas/tablaS_sitio_codigo_por_archivo.csv'],
        'S18': ['tablas/tablaS_rezago_registro.csv', 'tablas/tablaS_flujo_registros.csv'],
        'S19': ['tablas/tablaS_tendencia_ampliaciones.csv', 'reproduccion/pk/exploracion_previa/estimaciones_ITS.csv'],
        'S21B': ['resultados/lugar_defuncion.json'],
        'S22': ['tablas/tablaS_lugar_proporciones_estandarizadas.csv'],
        'S23': ['tablas/tablaS_lugar_proporcion_ajustada_anual.csv', 'tablas/tablaS_tasas_por_lugar_anuales.csv'],
    }
    catalogo = {}
    for fila in matriz[matriz.elemento.str.startswith('Tabla ')].to_dict('records'):
        num = fila['elemento'].removeprefix('Tabla ')
        estado, nota = 'recalculado', ''
        if num in fuente_directa:
            fuentes = [salida / fuente_directa[num]]
        elif num in guardadas:
            fuentes = [datos / 'benchmark/entradas' / guardadas[num]]
            estado = ESTADOS[1]
            nota = fila['motivo_si_bloqueado']
        elif num in ['S13','S16']:
            fuentes = [repo / 'notebooks/documentos_fuente.json']
            estado, nota = 'documental', 'Contenido descriptivo incluido en el inventario del notebook; no deriva de un cálculo.'
        else:
            # Todas las fuentes resultantes se redirigen a esta ejecución, nunca a REF.
            partes = fila['datos_entrada'].split(';')
            fuentes = []
            for parte in partes:
                parte = parte.strip()
                if parte.startswith('results/reference/'):
                    rel = parte.removeprefix('results/reference/')
                    modulo, resto = rel.split('/', 1)
                    fuentes.append(salida / modulo / modulo / resto)
            if not fuentes:
                raise ValueError(f'No hay fuente de ejecución para {num}')
        if num in fuentes_especiales:
            fuentes = [salida / 'c50/c50' / rel for rel in fuentes_especiales[num]]
        if num == 'S4':
            nota = 'Agregación recalculada; el entrenamiento LoRA y el ajuste completo proceden de archivos guardados.'
        if num == 'S14':
            nota = 'Compara dos juegos de pronósticos guardados; no vuelve a entrenarlos.'
        catalogo[num] = {'titulo': fila['descripcion'], 'estado': estado, 'nota': nota,
                        'fuentes': fuentes, 'codigo': fila['script'], 'especial': num in especiales}
    return catalogo


def mostrar_tabla(repo, salida, datos, catalogo, numero, plegada=False):
    entrada = catalogo[numero]
    if entrada['especial']:
        tabla = tabla_especial(numero, repo, salida, datos)
    else:
        tabla = pd.read_csv(entrada['fuentes'][0]).fillna('')
    exportada = salida / 'tablas_notebook' / f'Tabla_{numero}.csv'
    tabla.to_csv(exportada, index=False)
    meta = metadatos_documentos(repo)['tablas'][numero]
    enlaces = ' · '.join(enlace(p) for p in entrada['fuentes'])
    cuerpo = (f'<p><b>{entrada["estado"]}</b>. {escape(entrada["nota"])}</p>'
              f'<p>Fuente: {enlaces} · {enlace(exportada, "Exportar CSV")}</p>'
              f'<p><small>Código: {escape(entrada["codigo"])} · {enlace(repo / "scripts/gmm.py", "Ejecutor")}</small></p>'
              f'<div style="overflow:auto;max-height:480px">'
              + tabla.to_html(index=False, escape=True, na_rep='—', float_format=lambda v: f'{v:.6g}') + '</div>')
    titulo = f'Tabla {numero}. {meta["titulo"]}'
    if plegada:
        contenido = f'<details id="tabla-{numero}"><summary><b>{escape(titulo)}</b></summary>{cuerpo}</details>'
    else:
        contenido = f'<h3 id="tabla-{numero}">{escape(titulo)}</h3>{cuerpo}'
    if meta.get('nota'):
        contenido += f'<p><small>{escape(meta["nota"])}</small></p>'
    display(HTML(contenido))
    return {'elemento': f'Tabla {numero}', 'estado': entrada['estado'], 'archivo': str(exportada.relative_to(salida)),
            'sha256': sha256(exportada), 'filas': len(tabla), 'fuentes': [str(p) for p in entrada['fuentes']],
            'codigo': entrada['codigo'], 'nota': entrada['nota']}


def cerrar_reproduccion(repo, salida, datos, antes, inventario):
    despues = instantanea_entradas(repo, datos)
    esperados = {f'Tabla {n}' for n in metadatos_documentos(repo)['tablas']}
    esperados |= {f'Figura {n}' for n in metadatos_documentos(repo)['figuras']}
    encontrados = [x['elemento'] for x in inventario]
    if len(encontrados) != len(set(encontrados)) or set(encontrados) != esperados:
        raise AssertionError(f'Inventario incompleto/duplicado: {esperados.symmetric_difference(encontrados)}')
    fallas = []
    detalles = []
    comparaciones = {}
    for modulo in ['benchmark', 'c50']:
        informe = cargar_json(salida / modulo / 'comparacion_referencia.json')[modulo]
        comparaciones[modulo] = pd.Series([f['estado'] for f in informe]).value_counts().to_dict()
        fallas += [f'{modulo}: {f["archivo"]}' for f in informe
                   if f['estado'] not in ['identico_bytes', 'identico_contenido', 'coincide_tolerancia']
                   and not f['estado'].endswith('_conocido_verificado')]
        for fila in informe:
            if fila['estado'] == 'difiere':
                detalles += detallar_diferencias(repo / 'results/reference' / modulo / fila['archivo'],
                                                 salida / modulo / modulo / fila['archivo'],
                                                 f'{modulo}/{fila["archivo"]}')
    registro = cargar_json(salida / 'registro_notebook.json')
    registro.update(fin_utc=datetime.now(timezone.utc).isoformat(),
                    datos_y_referencias_sin_cambios=antes == despues,
                    inventario=inventario, comparaciones=comparaciones, discrepancias=fallas,
                    comparacion_estricta_coincide=not fallas)
    registro.pop('hashes_documentos', None)
    guardar_json(salida / 'registro_notebook.json', registro)
    pd.DataFrame(inventario).to_csv(salida / 'inventario_reproduccion.csv', index=False)
    pd.DataFrame(detalles).to_csv(salida / 'discrepancias_detalladas.csv', index=False)
    if antes != despues:
        raise AssertionError('Se modificaron entradas o referencias.')
    display(pd.DataFrame([{'Módulo': k, **v} for k, v in comparaciones.items()]).fillna(0))
    display(pd.DataFrame(inventario)[['elemento', 'estado']].groupby('estado', sort=False).count().rename(columns={'elemento':'Productos presentados'}))
    print(f'Cobertura completa: {len(inventario)} productos. Entradas y referencias intactas.')
    print('Las excepciones conocidas se verificaron con las reglas existentes; no se ampliaron tolerancias.')
    if fallas:
        display(HTML('<p><b>La comparación estricta contiene discrepancias:</b> '
                     + escape('; '.join(fallas)) + '. No se cuentan como coincidencias.</p>'))
        detalle = pd.DataFrame(detalles)
        if not detalle.empty:
            detalle['tipo_campo'] = detalle.campo.str.rsplit('/').str[-1]
            display(detalle.groupby(['archivo', 'tipo_campo']).agg(
                diferencias=('campo', 'size'), diferencia_absoluta_maxima=('diferencia_absoluta', 'max')).reset_index())
        print('Consulte discrepancias_detalladas.csv: conserva cada campo y los dos valores, con las tolerancias originales.')
    else:
        print('Comparación estricta: todos los archivos coinciden o tienen una excepción conocida verificada.')
    return registro


def detallar_diferencias(referencia, ejecucion, archivo):
    """Detalla todas las diferencias sin truncarlas ni cambiar el comparador original."""
    sys.path.insert(0, str(raiz_repositorio() / 'scripts'))
    from comparar_resultados import comparar_json, leer_csv
    filas = []
    def comparar(a, b, campo):
        if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
            for clave in sorted(a):
                comparar(a[clave], b[clave], campo + '/' + clave)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                comparar(x, y, campo + f'/{i}')
        elif comparar_json(a, b, 1e-9, 1e-12)['fallas']:
            filas.append({'archivo':archivo, 'campo':campo,
                          'referencia':b, 'ejecucion':a,
                          'diferencia_absoluta': abs(a-b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None})
    if str(referencia).endswith('.json'):
        comparar(cargar_json(ejecucion), cargar_json(referencia), '')
    else:
        a, b = leer_csv(ejecucion), leer_csv(referencia)
        if len(a) != len(b) or a[0] != b[0]:
            filas.append({'archivo':archivo, 'campo':'estructura CSV', 'referencia':str(b[0]), 'ejecucion':str(a[0])})
        else:
            for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), 1):
                for col, x, y in zip(a[0], ra, rb):
                    if x == y:
                        continue
                    try:
                        x, y = float(x), float(y)
                    except ValueError:
                        pass
                    comparar(x, y, f'fila {i}/{col}')
    return filas


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '_rotulos':
        raise SystemExit('Este módulo se importa desde el notebook.')
    _rotulos_portables(*(Path(p).resolve() for p in sys.argv[2:]))
