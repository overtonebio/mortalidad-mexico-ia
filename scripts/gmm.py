#!/usr/bin/env python3
"""Ejecutor por niveles del código del estudio GMM v13 (mortalidad esperada en México; cáncer de mama).

Niveles (véase docs/REPRODUCCION.md):
  1  verificar            integridad SHA-256, formatos, dimensiones y valores finitos de entradas y referencias
  2  figuras              regenera las 12 figuras desde resultados guardados (o desde una ejecución de nivel 3)
  3  benchmark | c50      recalcula métricas, intervalos, detección, tablas y análisis C50 desde las entradas
  4  entrenamiento        solo documenta; no ejecuta inferencia de modelos ni entrenamiento
  -  comparar             compara una ejecución con results/reference (y figuras con los PDF v13, si se indican)
  -  verificar-archivo    verifica una copia íntegra del archivo científico v13 original con su manifiesto

Los resultados de referencia (results/reference) nunca se modifican: cada ejecución trabaja en una copia
bajo --output-dir, que debe ser un directorio nuevo o vacío. No hay descargas ni entrenamiento implícitos.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / 'configs'
REF = REPO / 'results' / 'reference'
SRC = REPO / 'src'


class ErrorUsuario(Exception):
    """Error con mensaje para la persona usuaria (sin traza)."""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cargar(nombre: str) -> dict:
    return json.loads((CONF / nombre).read_text(encoding='utf-8'))


def _dentro(a: Path, b: Path) -> bool:
    try:
        a.resolve().relative_to(b.resolve()); return True
    except ValueError:
        return False


def preparar_salida(out: Path, data_dir: Path | None = None) -> Path:
    out = out.expanduser()
    if _dentro(out, REF) or out.resolve() == REPO.resolve() or _dentro(REPO, out):
        raise ErrorUsuario(f'--output-dir no puede ser el repositorio, contenerlo ni estar en results/reference: {out}')
    if data_dir is not None and (_dentro(out, data_dir) or _dentro(data_dir, out)):
        raise ErrorUsuario('--output-dir y --data-dir deben ser directorios separados')
    if out.exists() and any(out.iterdir()):
        raise ErrorUsuario(f'--output-dir ya contiene archivos: {out}. Use un directorio nuevo; las ejecuciones no se sobrescriben.')
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def verificar_entradas(data_dir: Path, subconjunto: str | None = None, rapido: bool = False) -> list[str]:
    """Comprueba presencia y SHA-256 de las entradas esperadas. Devuelve lista de problemas."""
    if not data_dir.is_dir():
        raise ErrorUsuario(f'--data-dir no existe: {data_dir}. Descomprima ahí los paquetes de datos y pronósticos (docs/DATOS.md).')
    esperados = cargar('datos_esperados.json')['archivos']
    problemas = []
    for rel, info in esperados.items():
        if subconjunto and not rel.startswith(subconjunto):
            continue
        p = data_dir / rel
        if not p.is_file():
            problemas.append(f'falta: {rel}')
        elif p.stat().st_size != info['bytes'] or (not rapido and sha256(p) != info['sha256']):
            problemas.append(f'alterado: {rel}')
    return problemas


def _entorno_hijo(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    # Los scripts originales importan módulos hermanos (p. ej., comun.py) desde su directorio.
    # PYTHONSAFEPATH=1 (o python -P) impide ese import; se retira solo para los procesos hijos.
    env.pop('PYTHONSAFEPATH', None)
    env.pop('ESTRATOS', None)
    env.setdefault('MPLBACKEND', 'Agg')
    if extra:
        env.update(extra)
    return env


def _versiones() -> dict:
    v = {'python': platform.python_version(), 'ejecutable': Path(sys.executable).name, 'plataforma': platform.platform(),
         'maquina': platform.machine(), 'procesador': platform.processor() or 'desconocido', 'cpu_logicas': os.cpu_count()}
    for m in ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib']:
        try:
            v[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001 - registro informativo
            v[m] = 'no instalado'
    return v


class Registro:
    def __init__(self, out: Path, nivel: str, args: argparse.Namespace):
        self.out, self.logs = out, out / 'logs'
        self.logs.mkdir(exist_ok=True)
        self.d = {'nivel': nivel, 'inicio_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'comando': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]), 'entorno': _versiones(),
                  'argumentos': {k: str(v) for k, v in vars(args).items() if k != 'func'}, 'pasos': []}

    def ejecutar(self, nombre: str, cmd: list[str], cwd: Path, env: dict) -> None:
        log = self.logs / f'{len(self.d["pasos"]) + 1:02d}_{nombre}.log'
        t0 = time.time()
        print(f'[{len(self.d["pasos"]) + 1:02d}] {nombre} ...', flush=True)
        with open(log, 'w', encoding='utf-8') as fh:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        paso = {'nombre': nombre, 'comando': [Path(c).name if i == 0 else c for i, c in enumerate(cmd)], 'cwd': str(cwd.relative_to(self.out)),
                'segundos': round(time.time() - t0, 2), 'codigo_salida': r.returncode, 'log': str(log.relative_to(self.out))}
        self.d['pasos'].append(paso)
        self.guardar()
        if r.returncode != 0:
            cola = log.read_text(encoding='utf-8', errors='replace').splitlines()[-15:]
            raise ErrorUsuario(f'El paso {nombre} terminó con código {r.returncode}. Últimas líneas de {log}:\n' + '\n'.join(cola))

    def guardar(self, **extra) -> None:
        self.d.update(extra)
        self.d['fin_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        (self.out / 'registro_ejecucion.json').write_text(json.dumps(self.d, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def _copiar_arbol(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))


def _hashes(raiz: Path, subdirs: list[str]) -> dict:
    return {str(p.relative_to(raiz)): sha256(p) for s in subdirs for p in sorted((raiz / s).rglob('*')) if p.is_file()}


# ----------------------------------------------------------------------------- nivel 1
def cmd_verificar(a: argparse.Namespace) -> int:
    fallas = []
    ref = json.loads((REF / 'SHA256SUMS.json').read_text())
    malos = [r for r, h in ref.items() if not (REF / r).is_file() or sha256(REF / r) != h]
    print(f'Resultados de referencia: {len(ref) - len(malos)}/{len(ref)} archivos íntegros')
    fallas += [f'referencia alterada o ausente: {r}' for r in malos]
    if a.data_dir:
        d = Path(a.data_dir).expanduser()
        prob = verificar_entradas(d)
        n = len(cargar('datos_esperados.json')['archivos'])
        print(f'Entradas en --data-dir: {n - len(prob)}/{n} archivos íntegros')
        fallas += prob
        fallas += _verificar_formatos(d)
    else:
        print('Sin --data-dir: solo se verificaron los resultados de referencia del repositorio.')
    fallas += _verificar_formatos_referencia()
    if a.informe:
        Path(a.informe).write_text(json.dumps({'fallas': fallas}, indent=1, ensure_ascii=False), encoding='utf-8')
    for f in fallas[:50]:
        print('FALLA', f)
    print('PASS' if not fallas else f'FAIL ({len(fallas)} problemas)')
    print('Nota: este nivel comprueba integridad y formato; no constituye validación científica.')
    return 0 if not fallas else 1


def _verificar_formatos(d: Path) -> list[str]:
    try:
        import numpy as np
    except ImportError:
        return ['numpy no instalado: no se verificaron dimensiones ni finitud de los NPZ']
    fallas, H = [], {'fc_2014-12': 36, 'fc_2015-12': 36, 'fc_2016-12': 36, 'fc_2019-12': 60, 'fc_h48_2014-12': 48, 'fc_h48_2015-12': 48}
    modelos = ['Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5', 'Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado', 'LightGBM global', 'N-HiTS global']
    n = 0
    for sub in ['publicados', 'nueva_ejecucion_semillas']:
        for nombre, h in H.items():
            p = d / 'benchmark/pronosticos' / sub / f'{nombre}.npz'
            if not p.is_file():
                continue
            with np.load(p, allow_pickle=False) as z:
                if sorted(z.files) != sorted(modelos):
                    fallas.append(f'{p.name} ({sub}): modelos {sorted(z.files)}')
                for m in z.files:
                    x = z[m]
                    if x.shape != (2870, h, 3):
                        fallas.append(f'{sub}/{p.name}/{m}: forma {x.shape}, esperada (2870, {h}, 3)')
                    if not np.isfinite(x).all():
                        fallas.append(f'{sub}/{p.name}/{m}: valores no finitos')
                    n += 1
    print(f'NPZ: {n} matrices revisadas (9 modelos × 12 archivos esperados = 108); el ensamble se calcula desde los miembros.')
    try:
        import pandas as pd
        c = pd.read_csv(d / 'benchmark/datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)
        y = c[[str(i) for i in range(156)]].to_numpy(float)
        if y.shape != (2870, 156) or not np.isfinite(y).all() or (y < 0).any():
            fallas.append(f'corpus2: forma {y.shape} o valores no finitos/negativos')
        else:
            print('Corpus: 2 870 series × 156 meses (2012-2024), finitos y no negativos.')
    except Exception as e:  # noqa: BLE001
        fallas.append(f'corpus2 no legible: {e}')
    return fallas


def _verificar_formatos_referencia() -> list[str]:
    fallas = []
    for p in sorted(REF.rglob('*.json')):
        try:
            json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            fallas.append(f'JSON inválido {p.relative_to(REF)}: {e}')
    return fallas


def cmd_verificar_archivo(a: argparse.Namespace) -> int:
    raiz = Path(a.archivo).expanduser()
    if not (raiz / 'benchmark').is_dir():
        raise ErrorUsuario('Indique la carpeta Analisis_y_pronosticos descomprimida del archivo científico v13.')
    man = cargar('manifiesto_archivo_cientifico_v13.json')['archivos']
    malos = [r for r, h in man.items() if not (raiz / r).is_file() or sha256(raiz / r) != h]
    print(('PASS' if not malos else 'FAIL'), f'{len(man)} archivos del manifiesto original; {len(malos)} ausentes o alterados')
    for r in malos[:50]:
        print('  ', r)
    return 0 if not malos else 1


# ----------------------------------------------------------------------------- nivel 3
def cmd_benchmark(a: argparse.Namespace) -> int:
    d = Path(a.data_dir).expanduser().resolve()
    prob = verificar_entradas(d, 'benchmark/')
    if prob:
        raise ErrorUsuario('Entradas del benchmark incompletas o alteradas:\n  ' + '\n  '.join(prob[:20]))
    out = preparar_salida(Path(a.output_dir), d)
    reg = Registro(out, '3-benchmark', a)
    w = out / 'benchmark'
    w.mkdir()
    _copiar_arbol(SRC / 'benchmark', w / 'codigo')
    for s in ['datos', 'entradas', 'pronosticos']:
        _copiar_arbol(d / 'benchmark' / s, w / s)
    shutil.copy2(CONF / 'manifiesto_pronosticos_sha256.txt', w / 'manifiesto_pronosticos_sha256.txt')
    for line in (w / 'manifiesto_pronosticos_sha256.txt').read_text().splitlines():
        esperado, rel = line.split(maxsplit=1)
        if sha256(w / 'pronosticos' / rel) != esperado:
            raise ErrorUsuario(f'Pronóstico alterado: {rel}')
    for s in ['resultados/benchmark', 'resultados/benchmark_nueva_ejecucion', 'documentos/tablas']:
        (w / s).mkdir(parents=True, exist_ok=True)
    entradas_antes = _hashes(w, ['datos', 'entradas', 'pronosticos'])
    cfg = cargar('pasos.json')['benchmark']
    for i, p in enumerate(cfg['pasos'], 1):
        nombre = Path(p['script']).stem + ('_' + '_'.join(f'{k}{v}' for k, v in p.get('env', {}).items()) if p.get('env') else '')
        reg.ejecutar(nombre, [sys.executable, p['script'], *p.get('args', [])], w, _entorno_hijo(p.get('env')))
    intactas = entradas_antes == _hashes(w, ['datos', 'entradas', 'pronosticos'])
    reg.guardar(entradas_sin_cambios=intactas)
    if not intactas:
        raise ErrorUsuario('Un paso modificó datos de entrada de la copia de trabajo; revise los registros.')
    print(f'Benchmark terminado en {out}. Compare con: python scripts/gmm.py comparar --output-dir {out}')
    return 0


def cmd_c50(a: argparse.Namespace) -> int:
    d = Path(a.data_dir).expanduser().resolve()
    prob = verificar_entradas(d, 'c50/')
    if prob:
        raise ErrorUsuario('Entradas C50 incompletas o alteradas:\n  ' + '\n  '.join(prob[:20]))
    out = preparar_salida(Path(a.output_dir), d)
    reg = Registro(out, '3-c50', a)
    w = out / 'c50'
    w.mkdir()
    _copiar_arbol(SRC / 'c50', w / 'codigo')
    for s in ['paquete_original', 'fuentes_inegi']:
        _copiar_arbol(d / 'c50' / s, w / s)
    for s in ['resultados', 'tablas', 'reproduccion', 'figuras']:
        (w / s).mkdir()
    entradas_antes = _hashes(w, ['paquete_original', 'fuentes_inegi'])
    cfg = cargar('pasos.json')['c50']
    env = dict(cfg['env'], NSIM=str(a.nsim), OPENBLAS_NUM_THREADS=str(a.hilos), OMP_NUM_THREADS=str(a.hilos))
    pasos = list(cfg['pasos'])
    if a.figuras:
        pasos.insert(pasos.index(cfg['paso_figuras']['despues_de']) + 1, cfg['paso_figuras']['script'])
    for s in pasos:
        reg.ejecutar(s, [sys.executable, f'{s}.py'], w / 'codigo', _entorno_hijo(env))
    intactas = entradas_antes == _hashes(w, ['paquete_original', 'fuentes_inegi'])
    reg.guardar(entradas_sin_cambios=intactas, nsim=a.nsim, hilos=a.hilos)
    if not intactas:
        raise ErrorUsuario('Un paso modificó datos de entrada de la copia de trabajo; revise los registros.')
    if a.nsim != 10000:
        print('Aviso: NSIM distinto de 10 000; los intervalos por simulación no serán comparables con la referencia.')
    print(f'C50 terminado en {out}. Compare con: python scripts/gmm.py comparar --output-dir {out}')
    return 0


# ----------------------------------------------------------------------------- nivel 2
def cmd_figuras(a: argparse.Namespace) -> int:
    d = Path(a.data_dir).expanduser().resolve()
    prob = verificar_entradas(d, 'benchmark/datos') + verificar_entradas(d, 'benchmark/entradas') + verificar_entradas(d, 'c50/')
    if prob:
        raise ErrorUsuario('Entradas necesarias para las figuras incompletas o alteradas:\n  ' + '\n  '.join(prob[:20]))
    fuente_b, fuente_c = REF / 'benchmark', REF / 'c50'
    if a.resultados_benchmark:
        fuente_b = Path(a.resultados_benchmark).expanduser().resolve() / 'benchmark'
        if not (fuente_b / 'resultados/benchmark/numeros_repro.json').is_file():
            raise ErrorUsuario('--resultados-benchmark debe ser el --output-dir de una ejecución completa de "gmm.py benchmark"')
    if a.resultados_c50:
        fuente_c = Path(a.resultados_c50).expanduser().resolve() / 'c50'
        if not (fuente_c / 'tablas/tablaS_tendencia_ampliaciones.csv').is_file():
            raise ErrorUsuario('--resultados-c50 debe ser el --output-dir de una ejecución completa de "gmm.py c50"')
    out = preparar_salida(Path(a.output_dir), d)
    reg = Registro(out, '2-figuras', a)
    t = out / 'trabajo'
    fb = t / 'benchmark'
    fb.mkdir(parents=True)
    for s in ['datos', 'entradas']:
        _copiar_arbol(d / 'benchmark' / s, fb / s)
    _copiar_arbol(fuente_b / 'resultados', fb / 'resultados')
    fc = t / 'c50'
    _copiar_arbol(SRC / 'c50', fc / 'codigo')
    _copiar_arbol(d / 'c50/paquete_original', fc / 'paquete_original')
    _copiar_arbol(fuente_c / 'resultados', fc / 'resultados')
    _copiar_arbol(fuente_c / 'tablas', fc / 'tablas')
    (fc / 'figuras').mkdir()
    f1 = t / 'fig1' / 'codigo' / 'documentos'
    f1.mkdir(parents=True)
    shutil.copy2(SRC / 'figuras/figura_diseno_v11.py', f1)
    env = _entorno_hijo()
    # figuras_c50.py lee resultados/c50_analitico.pkl, que no se distribuye: se reconstruye con
    # validacion_datos.py en una copia aparte, para no reescribir las tablas de referencia copiadas.
    fp = t / 'c50_base_analitica'
    _copiar_arbol(SRC / 'c50', fp / 'codigo')
    for s_ in ['paquete_original', 'fuentes_inegi']:
        _copiar_arbol(d / 'c50' / s_, fp / s_)
    for s_ in ['resultados', 'tablas']:
        (fp / s_).mkdir()
    reg.ejecutar('c50_base_analitica', [sys.executable, 'validacion_datos.py'], fp / 'codigo', env)
    shutil.copy2(fp / 'resultados/c50_analitico.pkl', fc / 'resultados/c50_analitico.pkl')
    reg.ejecutar('figura_1', [sys.executable, str(f1 / 'figura_diseno_v11.py')], t / 'fig1', env)
    reg.ejecutar('figuras_2_3_4_S1_S2', [sys.executable, str(SRC / 'figuras/exportar_figuras_v13.py'), '--salida', str(t / 'benchmark_pdf')], fb, env)
    reg.ejecutar('figuras_6_S3', [sys.executable, str(SRC / 'figuras/actualizar_figuras.py'), '--benchmark', str(fb), '--out', str(t / 'v13')], fb, env)
    reg.ejecutar('figuras_5_S4_S5_S6', [sys.executable, 'figuras_c50.py'], fc / 'codigo', env)
    final = out / 'figuras'
    final.mkdir()
    mapa = cargar('figuras.json')['figuras']
    for nombre, info in mapa.items():
        src = t / info['archivo_generado']
        if not src.is_file():
            raise ErrorUsuario(f'No se generó {info["archivo_generado"]} para {nombre}')
        shutil.copy2(src, final / f'{nombre}.pdf')
    reg.guardar(origen_resultados={'benchmark': 'results/reference' if not a.resultados_benchmark else 'ejecución nivel 3',
                                   'c50': 'results/reference' if not a.resultados_c50 else 'ejecución nivel 3'})
    print(f'12 figuras en {final}. Compárelas con: python scripts/gmm.py comparar --output-dir {out} --figuras-v13 <carpeta con Figura_*.pdf>')
    print('Nivel 2: regeneración desde resultados guardados; no es una reproducción completa.')
    return 0


# ----------------------------------------------------------------------------- comparación
def cmd_comparar(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from comparar_resultados import comparar, resumen
    from excepciones_comparacion import comprobar_excepcion
    out = Path(a.output_dir).expanduser().resolve()
    if not out.is_dir():
        raise ErrorUsuario('--output-dir no existe; indique una ejecución terminada.')
    if not any((out / n).is_dir() for n in ['benchmark', 'c50', 'figuras']):
        raise ErrorUsuario('No hay resultados benchmark, c50 ni figuras para comparar en --output-dir.')
    if not (a.rtol >= 0 and a.atol >= 0) or not all(__import__('math').isfinite(v) for v in [a.rtol, a.atol]):
        raise ErrorUsuario('Las tolerancias deben ser finitas y no negativas.')
    cfg = cargar('pasos.json')
    informe, fallas = {}, 0
    for mod in ['benchmark', 'c50']:
        w = out / mod
        if not w.is_dir():
            continue
        rels = []
        for s in cfg[mod]['salidas_comparables']:
            rels += [str(p.relative_to(w)) for p in sorted((w / s).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
        rels += [str(p.relative_to(REF / mod)) for s in cfg[mod]['salidas_comparables'] for p in sorted((REF / mod / s).rglob('*')) if p.is_file()]
        rels = sorted(set(rels) - set(cfg[mod].get('no_distribuidos', [])))
        filas = comparar(REF / mod, w, rels, a.rtol, a.atol)
        conocidas = cargar('diferencias_conocidas.json').get(mod, {})
        for f in filas:
            regla = conocidas.get(f['archivo'])
            if regla and comprobar_excepcion(f, regla, REF / mod / f['archivo'], w / f['archivo'], a.rtol, a.atol):
                f['estado'] += '_conocido_verificado'; f['nota_conocida'] = regla['nota']
        informe[mod] = filas
        r = resumen(filas)
        print(mod, r)
        fallas += r.get('difiere', 0) + r.get('no_generado', 0) + r.get('difiere_bytes_sin_comparador', 0) + r.get('sin_referencia', 0)
    fig = out / 'figuras'
    val = out / 'trabajo/v13/validacion_datos_figuras.json'
    if val.is_file():
        igual = sha256(val) == sha256(REF / 'figuras/validacion_datos_figuras.json')
        informe['huella_datos_figuras_6_S3'] = {'identica_a_referencia': igual}
        print('huella de datos de artistas (figuras 6 y S3):', 'idéntica a la referencia' if igual else
              'difiere (esperable si numpy/pandas no son los de environments/c50-figuras.txt; compare el render)')
    if fig.is_dir():
        esperadas = cargar('figuras.json')['figuras']
        faltantes = [f'{n}.pdf' for n in esperadas if not (fig / f'{n}.pdf').is_file()]
        extras = [p.name for p in fig.glob('*.pdf') if p.stem not in esperadas]
        informe['inventario_figuras'] = {'esperadas': len(esperadas), 'faltantes': faltantes, 'extras': extras}
        fallas += len(faltantes) + len(extras)
        if faltantes or extras:
            print('Inventario de figuras incompleto o inesperado:', informe['inventario_figuras'])
        from comparar_pdf import comparar_pdf
        if a.figuras_v13:
            try:
                import pypdfium2  # noqa: F401
            except ImportError:
                raise ErrorUsuario('Para comparar figuras instale environments/comparacion-figuras.txt (pypdfium2).')
            v13 = Path(a.figuras_v13).expanduser()
            filas = []
            for p in sorted(fig.glob('Figura_*.pdf')):
                cand = list(v13.rglob(p.name))
                filas.append({'archivo': p.name, **(comparar_pdf(p, cand[0]) if cand else {'estado': 'sin_pdf_v13'})})
            informe['figuras'] = filas
            ok = sum(1 for f in filas if f.get('pix_dif_frac') == 0.0 and f.get('texto_igual'))
            print(f'figuras: {ok}/{len(filas)} idénticas al render de 150 ppp y con el mismo texto')
            fallas += len(filas) - ok
        else:
            mapa = cargar('figuras.json')['figuras']
            filas = [{'archivo': f'{n}.pdf', 'sha256_igual_v13': sha256(fig / f'{n}.pdf') == i.get('sha256_pdf_v13')} for n, i in mapa.items() if (fig / f'{n}.pdf').is_file()]
            informe['figuras_sha256'] = filas
            print('figuras: sin --figuras-v13 solo se compara SHA-256 (los PDF incluyen fecha de creación; se espera que difieran).')
    (out / 'comparacion_referencia.json').write_text(json.dumps(informe, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Informe: {out / "comparacion_referencia.json"}')
    return 0 if fallas == 0 else 1


def cmd_entrenamiento(a: argparse.Namespace) -> int:
    print((REPO / 'docs' / 'ENTRENAMIENTO.md').read_text(encoding='utf-8'))
    print('\nEste comando no ejecuta inferencia ni entrenamiento. Siga docs/ENTRENAMIENTO.md en un entorno separado.')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='gmm.py', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    v = s.add_parser('verificar', help='nivel 1: integridad, formatos, dimensiones y finitud')
    v.add_argument('--data-dir', help='carpeta con los paquetes de datos y pronósticos descomprimidos')
    v.add_argument('--informe', help='ruta opcional para un informe JSON')
    v.set_defaults(func=cmd_verificar)
    va = s.add_parser('verificar-archivo', help='verifica una copia del archivo científico v13 original')
    va.add_argument('--archivo', required=True, help='carpeta Analisis_y_pronosticos descomprimida')
    va.set_defaults(func=cmd_verificar_archivo)
    for nombre, func, ayuda in [('benchmark', cmd_benchmark, 'nivel 3: recálculo del benchmark desde pronósticos guardados'),
                                ('c50', cmd_c50, 'nivel 3: análisis exploratorios C50 desde el extracto de microdatos')]:
        x = s.add_parser(nombre, help=ayuda)
        x.add_argument('--data-dir', required=True)
        x.add_argument('--output-dir', required=True, help='directorio nuevo o vacío para esta ejecución')
        if nombre == 'c50':
            x.add_argument('--nsim', type=int, default=10000, help='extracciones de simulación (referencia: 10 000)')
            x.add_argument('--hilos', type=int, default=2, help='hilos BLAS/OpenMP (referencia: 2)')
            x.add_argument('--figuras', action='store_true', help='incluye figuras_c50.py (figuras 5 y S4-S6)')
        x.set_defaults(func=func)
    f = s.add_parser('figuras', help='nivel 2: regenera las 12 figuras')
    f.add_argument('--data-dir', required=True)
    f.add_argument('--output-dir', required=True)
    f.add_argument('--resultados-benchmark', help='--output-dir de "gmm.py benchmark" (por omisión: results/reference)')
    f.add_argument('--resultados-c50', help='--output-dir de "gmm.py c50" (por omisión: results/reference)')
    f.set_defaults(func=cmd_figuras)
    c = s.add_parser('comparar', help='compara una ejecución con results/reference')
    c.add_argument('--output-dir', required=True)
    c.add_argument('--figuras-v13', help='carpeta con los PDF Figura_*.pdf del envío v13 para comparar por render')
    c.add_argument('--rtol', type=float, default=1e-9)
    c.add_argument('--atol', type=float, default=1e-12)
    c.set_defaults(func=cmd_comparar)
    e = s.add_parser('entrenamiento', help='nivel 4: muestra la documentación; no ejecuta')
    e.set_defaults(func=cmd_entrenamiento)
    a = p.parse_args(argv)
    try:
        return a.func(a)
    except ErrorUsuario as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
