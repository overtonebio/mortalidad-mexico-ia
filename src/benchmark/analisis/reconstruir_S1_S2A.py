"""Reconstrucción independiente de S2A y conteos de S1 desde corpus y NPZ guardados.

Creado en la revisión de Codex; no es el generador histórico recuperado. No entrena modelos.
S2A: orígenes 2014/2015, horizonte 48 meses, WQL y cobertura por año; rango por serie del
WQL medio de ambos orígenes; conformal al cuarto año, calibración 2014 y prueba 2015,
estratos de volumen históricos y cuantil finito de 90 % (method='higher').
S1: recalcula los conteos; conserva Tipo/CIE10 de la entrada como definición editorial.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from gmm import ErrorUsuario, preparar_salida, verificar_entradas
from comparar_resultados import comparar_csv

MIEMBROS = ['Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5']


def reconstruir(datos, salida):
    problemas = verificar_entradas(datos, 'benchmark/')
    if problemas:
        raise ErrorUsuario('Entradas incompletas o alteradas: ' + '; '.join(problemas[:10]))
    salida = preparar_salida(salida, datos)
    b = datos / 'benchmark'
    corpus = pd.read_csv(b / 'datos/corpus2_mortalidad_mensual.csv.gz', index_col=0)
    y = corpus[[str(i) for i in range(156)]].to_numpy(float)
    vol = pd.cut(corpus.mean_pre, [0, 10, 25, 100, 300, 1000, 1e7], labels=False).to_numpy()
    perdidas, cobertura, conjuntos, errores = {}, {}, {}, {}
    for origen, inicio in [('2014-12', 36), ('2015-12', 48)]:
        with np.load(b / f'pronosticos/publicados/fc_h48_{origen}.npz', allow_pickle=False) as archivo:
            forecasts = dict(archivo)
        forecasts['Ensamble (mediana)'] = np.median(np.stack([forecasts[m] for m in MIEMBROS]), axis=0)
        obs = y[:, inicio:inicio + 48]
        for modelo, f in forecasts.items():
            residual = obs[:, :, None] - f
            q = np.array([0.1, 0.5, 0.9])
            costo = np.maximum(q * residual, (q - 1) * residual)
            conjuntos.setdefault(modelo, []).append(2 * costo.sum(axis=(1, 2)) / (3 * np.abs(obs).sum(axis=1) + 1e-9))
            errores[modelo, origen] = np.log((obs[:, 36:48].sum(axis=1) + 1) / (f[:, 36:48, 1].sum(axis=1) + 1))
            for k in range(4):
                sl = slice(k * 12, (k + 1) * 12)
                perdidas.setdefault((modelo, k), []).append(2 * costo[:, sl].sum(axis=(1, 2)) / (3 * np.abs(obs[:, sl]).sum(axis=1) + 1e-9))
                cobertura.setdefault((modelo, k), []).append(((obs[:, sl] >= f[:, sl, 0]) & (obs[:, sl] <= f[:, sl, 2])).mean(axis=1) * 100)
    rangos = pd.DataFrame({m: np.mean(v, axis=0) for m, v in conjuntos.items()}).rank(axis=1).mean().sort_values()
    rows = []
    for m in rangos.index:
        row = {'Modelo': m}
        for k in range(4):
            relativo = np.concatenate(perdidas[m, k]) / np.concatenate(perdidas['Log-lineal', k])
            row[f'año {k + 1}'] = round(float((np.exp(np.log(relativo).mean()) - 1) * 100), 1)
        relativo = np.concatenate(conjuntos[m]) / np.concatenate(conjuntos['Log-lineal'])
        row['0-48'] = round(float((np.exp(np.log(relativo).mean()) - 1) * 100), 1)
        for k in range(4):
            row[f'cov_h{k + 1}'] = round(float(np.concatenate(cobertura[m, k]).mean()), 1)
        row['rango_serie'] = round(float(rangos[m]), 2)
        cal, test = np.abs(errores[m, '2014-12']), np.abs(errores[m, '2015-12'])
        umbrales = np.zeros(len(corpus))
        for v in np.unique(vol):
            grupo = cal[vol == v]
            umbrales[vol == v] = np.quantile(grupo, min(1, np.ceil((len(grupo) + 1) * 0.9) / len(grupo)), method='higher')
        row['FPR_año4'] = round(float((test > umbrales).mean() * 100), 1)
        row['semiancho_año4'] = round(float(np.median((np.exp(umbrales) - 1) * 100)), 1)
        rows.append(row)
    s2a = salida / 'tablaS2A_validacion48_benchmark.csv'
    pd.DataFrame(rows).to_csv(s2a, index=False)
    comparacion = comparar_csv(s2a, b / 'entradas/tablaS2A_validacion48_benchmark.csv', 0, 0)
    # S1: los códigos y tipos son definiciones conservadas, no inferidas de los conteos.
    s1 = pd.read_csv(b / 'entradas/tablaS1_grupos_CIE10.csv').set_index('Grupo')
    counts = corpus.groupby(['grp', 'level']).size().unstack(fill_value=0)
    for nivel in counts.columns:
        s1[nivel] = counts[nivel].reindex(s1.index)
    s1['Total series'] = s1[list(counts.columns)].sum(axis=1)
    s1 = s1.reset_index()[['Tipo', 'Grupo', 'CIE10', 'Nacional', 'Nacional×edad', 'Estatal', 'Estatal×edad', 'Total series']]
    s1path = salida / 'tablaS1_grupos_CIE10.csv'
    s1.to_csv(s1path, index=False)
    comparacion_s1 = comparar_csv(s1path, b / 'entradas/tablaS1_grupos_CIE10.csv', 0, 0)
    informe = {'S2A': {'filas': len(rows), 'valores_numericos': 120, 'comparacion': comparacion},
               'S1': {'series': int(len(corpus)), 'valores_conteo': 140, 'comparacion': comparacion_s1,
                      'alcance': 'Conteos recalculados; Tipo y CIE10 conservados como definiciones de entrada.'},
               'procedencia': 'Reconstrucción independiente de Codex cotejada contra las entradas históricas; no es código histórico recuperado.'}
    (salida / 'validacion_S1_S2A.json').write_text(json.dumps(informe, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ok = not comparacion['fallas'] and not comparacion_s1['fallas']
    print('PASS' if ok else 'FAIL', json.dumps(informe, ensure_ascii=False))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    a = p.parse_args()
    try:
        return reconstruir(a.data_dir.expanduser().resolve(), a.output_dir)
    except ErrorUsuario as e:
        print('ERROR:', e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
