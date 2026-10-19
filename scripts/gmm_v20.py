#!/usr/bin/env python3
"""Reproducción de la actualización v20: localidad, numeración y figuras del 17-oct-2026.

Reutiliza las entradas y los pronósticos v13 sin modificarlos. No descarga ni entrena modelos.
El comando figuras genera las doce figuras vigentes; localidad recalcula la ampliación ajustada.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
V20 = REPO / 'src/v20'


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def within(child, parent):
    return Path(child).resolve().is_relative_to(Path(parent).resolve())


def output_path(path, protected):
    out = Path(path).expanduser().resolve()
    for p in protected:
        if within(out, p) or within(p, out):
            raise ValueError(f'La salida debe quedar separada del código, los datos y la adenda: {out}')
    if out.exists():
        raise ValueError(f'La salida ya existe; use una carpeta nueva: {out}')
    return out


def verify_manifest(root, entries):
    root = Path(root).resolve()
    failures = []
    for rel, expected in entries.items():
        p = root / rel
        if not within(p, root):
            failures.append(f'ruta fuera del paquete: {rel}')
        elif not p.is_file() or sha256(p) != expected:
            failures.append(f'ausente o alterado: {rel}')
    if failures:
        raise ValueError('\n'.join(failures))
    return len(entries)


def verify_update():
    cfg = json.loads((REPO / 'configs/actualizacion_v20.json').read_text())
    return verify_manifest(REPO, cfg['archivos'])


def verify_adenda(path):
    root = Path(path).expanduser().resolve()
    spec = json.loads((root / 'MANIFEST_SHA256.json').read_text())
    return verify_manifest(root, {e['path']: e['sha256'] for e in spec['files']})


def run(cmd, cwd, log=None):
    env = os.environ.copy()
    env.pop('PYTHONSAFEPATH', None)
    env['MPLBACKEND'] = 'Agg'
    env['OPENBLAS_NUM_THREADS'] = '2'
    env['OMP_NUM_THREADS'] = '2'
    print('Paso:', ' '.join(Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(cmd)), flush=True)
    if log:
        with Path(log).open('w') as f:
            subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='command', required=True)
    v = s.add_parser('verificar', help='Verifica código v20, adenda y entradas v13')
    v.add_argument('--data-dir')
    v.add_argument('--adenda', type=Path)
    l = s.add_parser('localidad', help='Recalcula localidad y comprueba el resultado')
    l.add_argument('--data-dir', type=Path, required=True)
    l.add_argument('--output-dir', type=Path, required=True)
    l.add_argument('--prepare-only', action='store_true')
    f = s.add_parser('figuras', help='Regenera las doce figuras v20')
    f.add_argument('--data-dir', type=Path, required=True)
    f.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        n = verify_update()
        print(f'Código v20: {n}/{n} archivos íntegros')
        if getattr(a, 'adenda', None):
            a.adenda = a.adenda.expanduser().resolve()
            n = verify_adenda(a.adenda)
            print(f'Adenda: {n}/{n} archivos íntegros')
        if a.command == 'verificar':
            cmd = [sys.executable, REPO / 'scripts/gmm.py', 'verificar']
            if a.data_dir:
                cmd += ['--data-dir', Path(a.data_dir).expanduser().resolve()]
            run(cmd, REPO)
            return 0
        a.data_dir = a.data_dir.expanduser().resolve()
        protected = [REPO, a.data_dir]
        out = output_path(a.output_dir, protected)
        if a.command == 'localidad':
            from comparar_resultados import comparar
            raw_rel = Path('revision_github_zenodo/auditoria_Codex/ejec_figuras/trabajo/c50/paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz')
            ref_rel = Path('Entrega_GitHub_Zenodo_GMM_v13_revisada_Codex/github/mortalidad-mexico-ia/results/reference/c50/resultados/v12_apvp_localidad/localidad_hogar.csv')
            raw = a.data_dir / 'c50/paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz'
            raw_expected = json.loads((REPO / 'configs/actualizacion_v20.json').read_text())['raw_localidad_sha256']
            if not raw.is_file() or sha256(raw) != raw_expected:
                raise ValueError('El extracto C50 para localidad falta o está alterado')
            work = out / 'trabajo/analisis_localidad'
            work.mkdir(parents=True)
            for f in ['interaccion_localidad.py', 'verificar_y_resumir.py', 'PLAN_20261005.txt']:
                shutil.copy2(V20 / 'localidad' / f, work / f)
            for rel, src in [(raw_rel, raw), (ref_rel, REPO / 'results/reference/c50/resultados/v12_apvp_localidad/localidad_hogar.csv')]:
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            (out / 'procedencia.json').write_text(json.dumps({'raw_sha256': sha256(raw), 'plan_sha256': sha256(work / 'PLAN_20261005.txt')}, indent=2) + '\n')
            if a.prepare_only:
                print('Preparación de localidad terminada sin ajustar modelos:', out)
                return 0
            run([sys.executable, work / 'interaccion_localidad.py', '--raw', out / raw_rel], out, out / '01_ajuste.log')
            run([sys.executable, work / 'verificar_y_resumir.py'], out, out / '02_verificacion.log')
            reference = REPO / 'results/reference/v20/localidad'
            rows = comparar(reference, work, sorted(p.name for p in reference.glob('*.csv')), 1e-8, 1e-10)
            (out / 'comparacion_v20.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
            if any(r['estado'] not in ['identico_bytes', 'identico_contenido', 'coincide_tolerancia'] for r in rows):
                raise ValueError('La comparación de localidad difiere de las referencias; consulte comparacion_v20.json')
            print('Localidad terminada:', out)
            return 0
        out.mkdir(parents=True)
        logs = out / 'logs'
        logs.mkdir()
        legacy = out / 'base_v13'
        run([sys.executable, REPO / 'scripts/gmm.py', 'figuras', '--data-dir', a.data_dir, '--output-dir', legacy], REPO, logs / '01_base.log')
        final = out / 'figuras'
        shutil.copytree(legacy / 'figuras', final)
        code = out / 'codigo_rotulos'
        shutil.copytree(REPO / 'src/figuras', code)
        for file in (V20 / 'figuras').glob('*.py'):
            shutil.copy2(file, code / file.name)
        benchmark = legacy / 'trabajo/benchmark'
        run([sys.executable, code / 'ejecutar_F3_rotulos_20261017.py', code, out / 'figura3'], benchmark, logs / '02_figura3.log')
        run([sys.executable, code / 'actualizar_figuras_rotulos_20261017.py', '--benchmark', benchmark, '--out', out / 'figura6_S3'], benchmark, logs / '03_figura6.log')
        c50 = legacy / 'trabajo/c50'
        shutil.copy2(code / 'figuras_c50_rotulos_20261017.py', c50 / 'codigo/figuras_c50_v20.py')
        run([sys.executable, c50 / 'codigo/figuras_c50_v20.py'], c50 / 'codigo', logs / '04_c50.log')
        run([sys.executable, code / 'ejecutar_S1_S2_v20.py', '--codigo', code, '--config', REPO / 'configs/cajas_S1_S2_v20.json', '--output-dir', out / 'figuras_S1_S2'], benchmark, logs / '05_S1_S2.log')
        changed = {
            'Figura_3': out / 'figura3/Figura_3.pdf',
            'Figura_5': c50 / 'figuras/Figura6_C50_tendencia_y_lugar.pdf',
            'Figura_6': out / 'figura6_S3/Figura_6.pdf',
            'Figura_S1': out / 'figuras_S1_S2/Figura_S1.pdf',
            'Figura_S2': out / 'figuras_S1_S2/Figura_S2.pdf',
            'Figura_S3': out / 'figura6_S3/Figura_S3.pdf',
            'Figura_S5': c50 / 'figuras/FiguraS_C50_lugar_anual_y_sensibilidades.pdf',
        }
        for name, src in changed.items():
            shutil.copy2(src, final / (name + '.pdf'))
        expected = json.loads((REPO / 'configs/figuras_v20.json').read_text())['figuras']
        if set(p.stem for p in final.glob('*.pdf')) != set(expected):
            raise ValueError('El inventario final no contiene exactamente las doce figuras esperadas')
        verify_update()
        (out / 'verificacion_v20.json').write_text(json.dumps({
            'manuscrito': 'v20', 'python': sys.version, 'figuras': {p.name: sha256(p) for p in final.glob('*.pdf')},
            'nota': 'Regeneración desde productos guardados. No equivale a reentrenar modelos. Comparar render y texto con los PDF del envío.'
        }, ensure_ascii=False, indent=2) + '\n')
        print('Doce figuras v20 en:', final)
        return 0
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
