"""Exporta en PDF vectorial las figuras 2, 3, 4, S1 y S2 del manuscrito v13.

Adaptación de exportar_figuras_pdf.py (paquete_GMM_SI_IA_v10_revision/codigo/documentos/, recuperado;
SHA-256 y diff en revision/). Cambios respecto del original, sin efecto sobre datos ni artistas:
  1. ejecuta figuras_base_v9.py (copia idéntica, byte a byte, de figs_v9.py del original);
  2. escribe en --salida en lugar de documentos/figuras_pdf;
  3. solo genera las cinco figuras vigentes en v13. Las figuras 1, 6 y S3 de v13 proceden de
     figura_diseno_v11.py y actualizar_figuras.py; sus versiones v9 no se generan para no confundirlas.
Ejecutar con el directorio de trabajo en una raíz con resultados/, entradas/ y datos/ (scripts/gmm.py lo prepara).
Requiere numpy, pandas y matplotlib.
"""
from pathlib import Path
import argparse
import sys
import json
import pandas as pd
import matplotlib

matplotlib.use('Agg')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from estilo_figuras import apply_figure_style

a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument('--salida', type=Path, required=True, help='directorio de salida de los PDF')
args = a.parse_args()
apply_figure_style()
g = {'__file__': str(BASE / 'figuras_base_v9.py')}
exec((BASE / 'figuras_base_v9.py').read_text(), g)
OUT = args.salida.resolve()
OUT.mkdir(parents=True, exist_ok=True)
N = json.loads(Path('resultados/benchmark/numeros_repro.json').read_text())
PV = N['plac_vol']
plac_m = pd.DataFrame({m: {'semiancho_%': PV[m]['semiancho_mediana'], 'FPR_%': PV[m]['FPR']} for m in PV}).T
nat = pd.read_csv('resultados/benchmark/atlas_repro.csv.gz')
nat = nat[nat.level == 'Nacional']
jobs = [
    ('fig1_desde_numeros', (), 'Figura_2.pdf'),
    ('fig2b', (plac_m, nat), 'Figura_3.pdf'),
    ('fig4b', (), 'Figura_4.pdf'),
    ('figS1b', (), 'Figura_S1.pdf'),
    ('figS2b', (), 'Figura_S2.pdf'),
]
for function, fargs, filename in jobs:
    g[function](*fargs, out=str(OUT / filename))
    print(OUT / filename)
