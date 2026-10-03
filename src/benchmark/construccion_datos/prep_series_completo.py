"""Series mensuales de cáncer de mama (C50), mujeres >= 20 años, 2012-2024: nacional, por edad, por región y por
derechohabiencia. Consolida prep_series.py y la celda interactiva 192 (regiones y derechohabiencia), que no estaba en
un script. Uso: python prep_series_completo.py <salida.csv>

Denominadores:
 - nacional y por edad: WPP 2024, mujeres por grupo quinquenal, 1 de julio, interpolados a mitad de mes;
 - regional: población nacional WPP de cada grupo quinquenal x proporción de ese grupo que residía en la región según
   el Censo de Población y Vivienda 2020 (INEGI, tabulado cpv2020_b_eum_01_poblacion.xlsx, hoja 02). La proporción
   regional se mantiene constante 2012-2024 (supuesto declarado).
Derechohabiencia (conteos, sin denominador): con seguridad social = códigos 2,3,4,5,6,10; sin seguridad social =
 1,7,9,12; resto (8, 11, 99 y faltantes) = no especificado, redistribuido proporcionalmente cada mes."""
import sys, numpy as np, pandas as pd
out_path = sys.argv[1]
d = pd.read_csv('art/c50_raw_2012_2024.csv.gz', dtype=str)
for c in ['sexo', 'edad', 'mes_ocurr', 'anio_ocur', 'anio_regis']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
d['age'] = np.where((d.edad >= 4000) & (d.edad < 4998), d.edad - 4000, np.nan)
w = d[(d.sexo == 2) & (d.age >= 20) & d.mes_ocurr.between(1, 12) & d.anio_ocur.between(2012, 2024)].copy()
w['date'] = pd.to_datetime(dict(year=w.anio_ocur, month=w.mes_ocurr, day=1))
w['ag'] = (np.minimum(w.age, 85) // 5 * 5).astype(int)
idx = pd.date_range('2012-01-01', '2024-12-01', freq='MS')
comp = (w[w.anio_ocur.between(2017, 2022)].assign(s=lambda x: x.anio_regis == x.anio_ocur).groupby('mes_ocurr').s.mean())
fac = pd.Series(1.0, idx); m24 = idx.year == 2024; fac[m24] = 1 / comp.reindex(idx[m24].month).values
cnt = w.groupby(['date', 'ag']).size().unstack(fill_value=0).reindex(idx, fill_value=0); cnt_adj = cnt.mul(fac, axis=0)
p = pd.read_csv('art/wpp_mex_female_age5.csv'); p['ag'] = np.minimum(p.AgeGrpStart, 85)
p = p[p.ag >= 20].groupby(['Time', 'ag']).PopFemale.sum().unstack() * 1000
p.index = pd.to_datetime(p.index.astype(str) + '-07-01')
pm = p.reindex(p.index.union(idx + pd.Timedelta(days=14))).interpolate('time').reindex(idx + pd.Timedelta(days=14)); pm.index = idx
who = pd.Series({20: 8220, 25: 7930, 30: 7610, 35: 7150, 40: 6590, 45: 6040, 50: 5370, 55: 4550, 60: 3720, 65: 2960, 70: 2210,
                 75: 1520, 80: 910, 85: 635}, dtype=float); who = who / who.sum()
rate = cnt_adj / (pm / 12) * 1e5; asmr = (rate[who.index] * who).sum(axis=1); crude = cnt_adj.sum(1) / (pm.sum(1) / 12) * 1e5
out = pd.DataFrame({'fecha': idx, 'deaths': cnt.sum(1).values, 'deaths_adj': cnt_adj.sum(1).values, 'pop20': pm.sum(1).values,
                    'crude_rate': crude.values, 'asmr': asmr.values})
for k, ags in {'20-39': [20, 25, 30, 35], '40-59': [40, 45, 50, 55], '60+': [60, 65, 70, 75, 80, 85]}.items():
    ww = who[ags] / who[ags].sum(); out[f'asmr_{k}'] = (rate[ags] * ww).sum(1).values; out[f'deaths_{k}'] = cnt_adj[ags].sum(1).values
# regiones
t = pd.read_excel('audit/censo/cpv2020_pob.xlsx', '02', header=None)
t2 = t.iloc[8:, :4].copy(); t2.columns = ['ent', 'sexo', 'edad', 'pop']
t2 = t2[(t2.sexo == 'Mujeres') & t2.edad.str.match(r'^\d\d-\d\d|^100', na=False) & (t2.ent != 'Estados Unidos Mexicanos')]
t2['ent_code'] = t2.ent.str[:2].astype(int); t2['a0'] = t2.edad.str[:3].str.extract(r'(\d+)')[0].astype(int); t2['ag'] = np.minimum(t2.a0, 85)
t2 = t2[t2.ag >= 20]; t2['pop'] = t2['pop'].astype(float)
REG = {'Norte': [2, 3, 5, 8, 10, 19, 25, 26, 28], 'Centro-Occidente': [1, 6, 11, 14, 16, 18, 22, 24, 32], 'Centro': [9, 13, 15, 17, 21, 29],
       'Sur-Sureste': [4, 7, 12, 20, 23, 27, 30, 31]}
e2r = {e: r for r, l in REG.items() for e in l}; assert len(e2r) == 32
t2['reg'] = t2.ent_code.map(e2r); share = t2.groupby(['reg', 'ag'])['pop'].sum().unstack(); share = share / share.sum()
w['ent'] = pd.to_numeric(w.ent_resid, errors='coerce'); w['reg'] = w.ent.map(e2r)
for r in REG:
    c = w[w.reg == r].groupby(['date', 'ag']).size().unstack(fill_value=0).reindex(index=idx, columns=who.index, fill_value=0).mul(fac, axis=0)
    popr = pm[who.index] * share.loc[r, who.index].values
    out[f'asmr_{r}'] = ((c / (popr / 12) * 1e5) * who).sum(1).values; out[f'deaths_{r}'] = c.sum(1).values
w['dh'] = pd.to_numeric(w.derechohab, errors='coerce')
w['ins'] = np.where(w.dh.isin([2, 3, 4, 5, 6, 10]), 'SS', np.where(w.dh.isin([1, 7, 9, 12]), 'noSS', 'NE'))
ic = w.groupby(['date', 'ins']).size().unstack(fill_value=0).reindex(idx, fill_value=0).mul(fac, axis=0); known = ic.SS + ic.noSS
out['deaths_SS'] = (ic.SS * (1 + ic.NE / known)).values; out['deaths_noSS'] = (ic.noSS * (1 + ic.NE / known)).values
out.to_csv(out_path, index=False)
print('sin región:', int(w.reg.isna().sum()), '| NE derechohabiencia:', round(float((ic.NE / ic.sum(1)).mean()) * 100, 2), '%')
print(share.round(4).to_string())
