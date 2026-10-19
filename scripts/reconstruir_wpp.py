#!/usr/bin/env python3
"""Reconstruye y coteja el extracto WPP usando el CSV oficial de julio, sin red.

Descargue FUENTE_URL; ejecute --fuente-csv-gz ARCHIVO --comparar-csv EXTRACTO
--output-dir CARPETA_NUEVA. Solo usa la biblioteca estándar de Python.
"""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

FUENTE_URL = ('https://population.un.org/wpp/assets/Excel%20Files/'
              '1_Indicator%20%28Standard%29/CSV_FILES/'
              'WPP2024_PopulationByAge5GroupSex_Medium.csv.gz')
CAMPOS = ['Time', 'AgeGrp', 'AgeGrpStart', 'PopFemale']

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clave(row):
    return int(row['Time']), int(row['AgeGrpStart'])

def comprobar_grano(rows):
    keys = [clave(row) for row in rows]
    expected = {(year, age) for year in range(2010, 2027) for age in range(0, 101, 5)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Se requieren 357 claves únicas: 2010-2026 × 21 grupos de edad.')
    for row in rows:
        age = int(row['AgeGrpStart'])
        group = '100+' if age == 100 else f'{age}-{age + 4}'
        value = Decimal(row['PopFemale'])
        if row['AgeGrp'] != group or not value.is_finite() or value <= 0:
            raise ValueError(f'Grupo o población inválida: {row}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fuente-csv-gz', type=Path, required=True)
    p.add_argument('--comparar-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('--output-dir debe ser una carpeta nueva.')
    rows = []
    with gzip.open(args.fuente_csv_gz, 'rt', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ISO3_code'] != 'MEX' or not 2010 <= int(row['Time']) <= 2026:
                continue
            if row['LocID'] != '484' or row['VarID'] != '2' or row['Variant'] != 'Medium':
                raise ValueError('País o variante inesperados.')
            if Decimal(row['MidPeriod']) != Decimal(row['Time']) + Decimal('0.5'):
                raise ValueError('La población debe corresponder al 1 de julio.')
            rows.append({key: row[key] for key in CAMPOS})
    rows.sort(key=clave)
    comprobar_grano(rows)
    with args.comparar_csv.open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    comprobar_grano(original)
    original.sort(key=clave)
    differences = []
    for source, historical in zip(rows, original):
        for key in CAMPOS:
            equal = (Decimal(source[key]) == Decimal(historical[key])
                     if key == 'PopFemale' else source[key] == historical[key])
            if not equal:
                differences.append({'clave': clave(source), 'campo': key,
                                    'fuente': source[key], 'historico': historical[key]})
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / 'wpp_mex_female_age5_reconstruido.csv'
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'fuente_url': FUENTE_URL,
        'fuente_archivo': args.fuente_csv_gz.name,
        'fuente_sha256': sha256(args.fuente_csv_gz),
        'fuente_bytes': args.fuente_csv_gz.stat().st_size,
        'licencia_fuente': 'CC-BY-3.0-IGO',
        'cita': 'United Nations, DESA, Population Division (2024). World Population Prospects 2024, Online Edition.',
        'filtro': 'ISO3_code=MEX; LocID=484; VarID=2 (Medium); Time=2010..2026',
        'transformacion': 'Seleccionar Time, AgeGrp, AgeGrpStart y PopFemale; ordenar por año y edad. Sin redondeo ni cambio de unidades.',
        'unidad': 'Miles de mujeres, población de facto al 1 de julio; grupo 100+ abierto.',
        'grano': 'Año × grupo quinquenal de edad; México, mujeres, variante media.',
        'filas': len(rows), 'claves_unicas': len({clave(x) for x in rows}),
        'comparacion_numerica_exacta': not differences,
        'diferencias': differences,
        'extracto_historico_sha256': sha256(args.comparar_csv),
        'reconstruccion_sha256': sha256(output),
        'bytes_identicos': output.read_bytes() == args.comparar_csv.read_bytes(),
        'alcance': 'Reconstrucción independiente; no identifica la descarga histórica original, cuyo registro no se conservó.'
    }
    (args.output_dir / 'verificacion_wpp.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if differences:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
