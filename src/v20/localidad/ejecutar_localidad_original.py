"""Envoltorio portable; conserva scripts y resultados de referencia inalterados."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RAW_LEGACY = Path('revision_github_zenodo/auditoria_Codex/ejec_figuras/trabajo/c50/paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz')
REF_LEGACY = Path('Entrega_GitHub_Zenodo_GMM_v13_revisada_Codex/github/mortalidad-mexico-ia/results/reference/c50/resultados/v12_apvp_localidad/localidad_hogar.csv')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True, help='Carpeta nueva fuera de esta adenda; nunca sobrescribe referencias.')
    p.add_argument('--prepare-only', action='store_true', help='Solo prepara y comprueba los archivos; no ejecuta modelos.')
    p.add_argument('--verify', action='store_true', help='Después del análisis, ejecuta también verificar_y_resumir.py.')
    p.add_argument('--deps', type=Path, help='Directorio opcional con dependencias YA instaladas; se antepone a PYTHONPATH.')
    a = p.parse_args()
    out = a.output_dir.expanduser().resolve()
    if out == HERE or HERE in out.parents:
        p.error('--output-dir debe quedar fuera de la adenda para preservar las referencias.')
    if out.exists():
        p.error('--output-dir ya existe; elija una carpeta nueva.')
    source = HERE/'localidad/codigo_original'
    work = out/'trabajo/analisis_localidad'
    work.mkdir(parents=True)
    copies=[]
    for name in ['interaccion_localidad.py','verificar_y_resumir.py','PLAN_20261005.txt']:
        dest=work/name; shutil.copy2(source/name,dest)
        assert digest(dest)==digest(source/name)
        copies.append({'file':str(dest.relative_to(out)),'sha256':digest(dest)})
    for relative, source_name in [(RAW_LEGACY,'c50_raw_2012_2024.csv.gz'),(REF_LEGACY,'localidad_hogar_v13.csv')]:
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        src=HERE/'localidad/datos'/source_name;shutil.copy2(src,dest)
        assert digest(dest)==digest(src)
        copies.append({'file':str(relative),'sha256':digest(dest)})
    (out/'preparacion.json').write_text(json.dumps({'copias_verificadas':copies,'referencias_preservadas':str(HERE/'localidad/referencia')},ensure_ascii=False,indent=2)+'\n')
    commands=[[sys.executable,str(work/'interaccion_localidad.py'),'--raw',str(out/RAW_LEGACY)]]
    if a.verify: commands.append([sys.executable,str(work/'verificar_y_resumir.py')])
    (out/'comandos.json').write_text(json.dumps(commands,ensure_ascii=False,indent=2)+'\n')
    if a.prepare_only:
        print('Preparado sin ejecutar modelos:',out)
        return
    env=os.environ.copy()
    if a.deps:
        deps=a.deps.expanduser().resolve()
        if not deps.is_dir(): p.error('--deps no existe o no es directorio.')
        env['PYTHONPATH']=str(deps)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for i,cmd in enumerate(commands,1):
        with (out/f'ejecucion_{i}.log').open('w') as log:
            subprocess.run(cmd,cwd=out,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    print('Finalizado. Resultados nuevos en:',work)
    print('Compare con las referencias de:',HERE/'localidad/referencia')


if __name__=='__main__': main()
