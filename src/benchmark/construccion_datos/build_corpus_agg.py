"""Corpus de series mensuales de mortalidad (INEGI EDR 2012-2024) para el benchmark de IA,
y base individual de mujeres >=20 años para el clasificador de subregistro."""
import zipfile, numpy as np, pandas as pd

RAW = 'inegi_raw'
COLS = ['ent_resid', 'tloc_resid', 'causa_def', 'sexo', 'edad', 'mes_ocurr', 'anio_ocur', 'escolarida',
        'edo_civil', 'derechohab', 'sitio_ocur', 'asist_medi', 'ocupacion']

def rng3(a, b):  # inclusive range of 3-char ICD codes with same letter
    L = a[0]; return {f'{L}{i:02d}' for i in range(int(a[1:]), int(b[1:]) + 1)}

GROUPS = {
    'Cardiopatía isquémica': rng3('I20', 'I25'), 'Enf. cerebrovascular': rng3('I60', 'I69'),
    'Enf. hipertensivas': rng3('I10', 'I15'), 'Diabetes mellitus': rng3('E10', 'E14'),
    'EPOC': rng3('J40', 'J44'), 'Influenza y neumonía': rng3('J09', 'J18'),
    'Enf. hepáticas': rng3('K70', 'K76'), 'Insuficiencia renal': rng3('N17', 'N19'),
    'Desnutrición': rng3('E40', 'E46'), 'VIH': rng3('B20', 'B24'),
    'Homicidio': rng3('X85', 'X99') | rng3('Y00', 'Y09'), 'Suicidio': rng3('X60', 'X84'),
    'Accidentes de tránsito': rng3('V01', 'V89'),
    'C. mama': {'C50'}, 'C. cérvix': {'C53'}, 'C. ovario': {'C56'}, 'C. próstata': {'C61'},
    'C. pulmón': {'C33', 'C34'}, 'C. colorrectal': rng3('C18', 'C21'), 'C. estómago': {'C16'},
    'C. hígado': {'C22'}, 'C. páncreas': {'C25'}, 'Leucemias': rng3('C91', 'C95'),
    'Linfomas y mieloma': rng3('C81', 'C90'), 'C. riñón': {'C64'}, 'Tumores SNC': rng3('C70', 'C72'),
    'Causas mal definidas': {f'R{i:02d}' for i in range(100)},
}
code2grp = {c: g for g, cs in GROUPS.items() for c in cs}

def main_member(z):
    big = [i for i in z.infolist() if i.file_size > 50e6 and 'bitacora' not in i.filename.lower()]
    assert len(big) == 1; return big[0].filename

agg, indiv = [], []
for y in range(2012, 2025):
    z = zipfile.ZipFile(f'{RAW}/edr_{y}.zip')
    with z.open(main_member(z)) as f:
        df = pd.read_csv(f, usecols=lambda c: c.replace('ï»¿', '').replace('\ufeff', '').strip() in COLS,
                         dtype=str, encoding='latin-1')
    df.columns = [c.replace('ï»¿', '').replace('\ufeff', '').strip() for c in df.columns]
    c3 = df.causa_def.str.upper().str.strip().str[:3]
    num = {c: pd.to_numeric(df[c], errors='coerce') for c in COLS if c != 'causa_def'}
    edad = num['edad']; age = np.where((edad >= 4000) & (edad < 4998), edad - 4000, np.where(edad < 4000, 0, np.nan))
    d = pd.DataFrame({'y': num['anio_ocur'], 'm': num['mes_ocurr'], 'sex': num['sexo'], 'ent': num['ent_resid'],
                      'age': age, 'c3': c3})
    d['grp'] = d.c3.map(code2grp)
    d['ab'] = pd.cut(d.age, [-1, 39, 64, 200], labels=['0-39', '40-64', '65+'])
    ok = d.y.between(2012, 2024) & d.m.between(1, 12) & d.sex.isin([1, 2])
    d = d[ok]
    allc = d.assign(grp='Todas las causas')
    dd = pd.concat([d[d.grp.notna()], allc])
    agg.append(dd.groupby(['grp', 'sex', 'ent', 'ab', 'y', 'm'], observed=True).size().rename('n').reset_index())
    # individual-level women >=20 (classifier)
    wmask = (num['sexo'] == 2) & (pd.Series(age) >= 20).values & num['anio_ocur'].between(2012, 2024) & num['mes_ocurr'].between(1, 12)
    wi = pd.DataFrame({'y': num['anio_ocur'], 'm': num['mes_ocurr'], 'age': age, 'c3': c3, 'ent': num['ent_resid'],
                       'tloc': num['tloc_resid'], 'esc': num['escolarida'], 'civ': num['edo_civil'],
                       'dh': num['derechohab'], 'sitio': num['sitio_ocur'], 'asist': num['asist_medi'],
                       'ocup': num['ocupacion'], 'file_year': y})[wmask.values]
    print(y, len(df), flush=True)

A = pd.concat(agg).groupby(['grp', 'sex', 'ent', 'ab', 'y', 'm'], observed=True).n.sum().reset_index()
A.to_csv('agg_mortalidad_raw.csv.gz', index=False)
