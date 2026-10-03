"""Reproduce APVP and descriptive locality results from retained C50 records. No network or sibling package is required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

M=Path(__file__).resolve().parents[1]
O=M/'resultados/v12_apvp_localidad'; O.mkdir(exist_ok=True, parents=True)
raw=pd.read_csv(M/'paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz',dtype=str)
for c in raw.columns: raw[c]=raw[c].str.strip()
for c in ['sexo','edad','mes_ocurr','anio_ocur','anio_regis']:
    raw[c]=pd.to_numeric(raw[c],errors='coerce')
raw['age']=raw.edad-4000
x=raw[(raw.sexo==2)&raw.edad.between(4020,4997)&raw.mes_ocurr.between(1,12)&raw.anio_ocur.between(2012,2024)].copy()
x['date']=pd.to_datetime(dict(year=x.anio_ocur,month=x.mes_ocurr,day=1))
x['ag']=np.minimum(x.age//5*5,85).astype(int)
x['apvp']=np.maximum(74.5-x.age,0)
ix=pd.date_range('2012-01-01','2024-12-01',freq='MS')
who=pd.Series([8220,7930,7610,7150,6590,6040,5370,4550,3720,2960,2210],index=range(20,75,5),dtype=float)
who=who/who.sum()
wpp=pd.read_csv(M/'paquete_original/editorial/datos/wpp_mex_female_age5.csv')
wpp['ag']=np.minimum(wpp.AgeGrpStart,85)
pop=wpp[wpp.ag>=20].groupby(['Time','ag']).PopFemale.sum().unstack()*1000
base=pop.copy();base.index=pd.to_datetime(base.index.astype(str)+'-07-01')
mid=ix+pd.Timedelta(days=14)
pm=base.reindex(base.index.union(mid)).interpolate('time').loc[mid];pm.index=ix
fac=pd.Series(1.,index=ix)
old=x[x.anio_ocur.between(2017,2022)]
complete=old.assign(same=old.anio_regis==old.anio_ocur).groupby('mes_ocurr').same.mean()
fac.loc['2024']=1/complete.reindex(ix[ix.year==2024].month).to_numpy()
apvp=x[x.age<75].groupby(['date','ag']).apvp.sum().unstack(fill_value=0).reindex(index=ix,columns=who.index,fill_value=0).mul(fac,axis=0)
rate=(apvp.div(pm[who.index]/12)*1e5*who).sum(axis=1)
annual=pd.DataFrame({'APVP_total':apvp.sum(axis=1).groupby(ix.year).sum(),
                    'APVP_tasa_estandarizada':rate.groupby(ix.year).mean(),
                    'defunciones':x.groupby('anio_ocur').size(),
                    'edad_media':x.groupby('anio_ocur').age.mean(),
                    'pct_menor_60':x.assign(young=x.age<60).groupby('anio_ocur').young.mean()*100})

def fit(idx):
    dates=ix[idx];t=((dates.year-2012)*12+dates.month-1).to_numpy()
    cols=[np.ones(len(t)),t/12,(t>=98).astype(float),np.maximum(t-98,0)/12]
    cols += [(dates.month==m).astype(float) for m in range(2,13)]
    X=np.array(cols).T;y=np.log(rate.loc[dates].to_numpy())
    b=np.linalg.lstsq(X,y,rcond=None)[0];u=y-X@b;n,k=X.shape
    score=X*u[:,None];meat=score.T@score
    for lag in range(1,13):
        matches=[(a,np.flatnonzero(t==tt-lag)[0]) for a,tt in enumerate(t) if (t==tt-lag).any()]
        if matches:
            a,z=map(np.array,zip(*matches));term=score[a].T@score[z]
            meat+=(1-lag/13)*(term+term.T)
    bread=np.linalg.inv(X.T@X);cov=bread@meat@bread*n/(n-k)
    out={}
    for name,terms in {'pre':[1],'nivel':[2],'cambio':[3],'post':[1,3]}.items():
        v=np.zeros(k);v[terms]=1;e=v@b;se=np.sqrt(v@cov@v)
        out[name]={'pct':100*np.expm1(e),'lo':100*np.expm1(e-1.95996398454*se),'hi':100*np.expm1(e+1.95996398454*se),'p':2*norm.sf(abs(e/se))}
    return out

fits={'principal_2012_2023':fit(ix.year<2024),'sin_2020_2012_2023':fit((ix.year<2024)&(ix.year!=2020)),
      'con_2024_corregido':fit(np.ones(len(ix),bool))}
a=x[(x.anio_ocur==2023)&(x.age<75)]
shares=a.groupby(pd.cut(a.age,[19,39,59,74])).apvp.sum()/a.apvp.sum()*100

# Locality descriptions use residents of Mexico and 2012-2023 occurrence.
for c in ['ent_resid','tloc_resid','sitio_ocur']:
    x[c]=pd.to_numeric(x[c],errors='coerce')
z=x[x.ent_resid.between(1,32)&(x.anio_ocur<=2023)].copy()
labels=['< 2 500','2 500-14 999','15 000-99 999','≥ 100 000']
z['localidad']=pd.cut(z.tloc_resid,[0,3,6,12,17],labels=labels)
rows=[]
for label in labels:
    for year in [2019,2020,2023]:
        d=z[(z.localidad==label)&(z.anio_ocur==year)]
        n=len(d);h=int((d.sitio_ocur==11).sum());unknown=int((d.sitio_ocur==99).sum())
        rows.append(dict(localidad=label,anio=year,n=n,hogar=h,lugar_no_especificado=unknown,
                         hogar_pct=100*h/n,hogar_pct_sin99=100*h/(n-unknown)))
local=pd.DataFrame(rows)
result={'N_C50':len(x),'N_localidad_universo':len(z),
        'localidad_no_clasificable':int(z.localidad.isna().sum()),
        'lugar99_todo_universo':int((z.sitio_ocur==99).sum()),
        'n_por_localidad_2012_2023':z.groupby('localidad',observed=True).size().to_dict(),
        'D_2023':annual.loc[2023].to_dict(),
        'D_cambio_2019_2023_pct':(annual.loc[2023]/annual.loc[2019]-1).mul(100).to_dict(),
        'APVP_edad2023_pct':{str(k):float(v) for k,v in shares.items()},'ITS':fits,
        'sensibilidad_localidad_max_pp':float((local.hogar_pct_sin99-local.hogar_pct).max()),
        'convenciones':{'APVP':'max(75 - edad cumplida - 0.5, 0); mujeres de 20-74 años',
            'tasa':'pesos OMS normalizados a 20-74; WPP2024 interpolada al día15; tasa mensual anualizada; media anual',
            '2024':'solo APVP y tasas corregidos por registro tardío según fracciones nacionales 2017-2022 por mes; muertes y edad descriptivas sin corregir',
            'ITS':'log tasa, tendencia en años, nivel y cambio pendiente marzo2020, indicadores mensuales; HAC Bartlett12 meses calendario n/(n-k); IC normal95',
            'localidad':'tloc_resid1-3 rural<2500;4-6 2500-14999;7-12 15000-99999;13-17 ≥100000;excluye localidad inválida',
            'hogar':'sitio_ocur11; denominador todos los lugares, incluido99; sensibilidad excluye99 únicamente',
            'plan':'Análisis posteriores a resultados originales. Sensibilidades APVP sin2020/con2024 y localidad sin99 añadidas en auditoría del 1oct2026.'}}
annual.to_csv(O/'APVP_anual.csv',index_label='anio')
rate.to_csv(O/'APVP_tasa_mensual.csv',index_label='fecha',header=['tasa'])
local.to_csv(O/'localidad_hogar.csv',index=False)
(O/'resumen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
