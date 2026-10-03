"""Valida v11; --baseline apunta opcionalmente a un paquete v10 solo para cotejo.
No modifica datos ni ajusta modelos. Escribe resultados/verificacion_v11.json.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from comun import ROOT, RAW, WPP, SERIES, poblacion_mensual, tee, IDX

a=argparse.ArgumentParser();a.add_argument('--baseline',type=Path);args=a.parse_args()
L=json.loads((ROOT/'resultados/lugar_defuncion.json').read_text())
T=json.loads((ROOT/'resultados/tendencia_ampliaciones.json').read_text())
N=json.loads((ROOT/'resultados/numeros_v11.json').read_text())
w=pd.read_pickle(ROOT/'resultados/c50_analitico.pkl')
base=w[w.residente & (w.anio_ocur<=2023)]
kn=base[base.sitio_ocur!=99]
v={'fecha':'2026-09-30','n_principal':len(base),'n_conocido':len(kn),
   'n_codigo99':int((base.sitio_ocur==99).sum()),
   'n_codigos10_12_conservados':int(kn.sitio_ocur.isin([10,12]).sum()),'comprobaciones':{}}
for site in ['Hogar','Unidad pública','Unidad privada']:
    assert L['sens_lugar_conocido_'+site]['n_defunciones']==len(kn)
unknown=L['sens_catalogo_alt_No especificado']
assert unknown['inferencial'] is False and unknown['diagnosticos']['separacion_estructural']
assert not set(['OR','periodos','wald_nivel_pendiente','coeficientes']).intersection(unknown)
assert 'catalogo_alt_No especificado' not in N['lugar_sens_2023']
OR=pd.read_csv(ROOT/'tablas/tablaS_lugar_razones_momios.csv')
assert not OR.desenlace.eq('No especificado').any()
assert N['modelos_no_inferenciales']['sens_catalogo_alt_No especificado']['inferencial'] is False
v['comprobaciones']['codigo99_sin_inferencia']=True
v['comprobaciones']['lugar_conocido_solo_excluye99']=True
TS=pd.read_csv(ROOT/'resultados/tasas_estandarizadas_por_lugar_mensual.csv',parse_dates=['fecha']).set_index('fecha')
count=w.groupby(['date','ag']).size().unstack(fill_value=0).reindex(IDX,fill_value=0)
expected=tee(count,poblacion_mensual())
v['tasas_observadas_max_abs']=float(np.max(np.abs(TS.Total.values-expected.values)))
v['descomposicion_tasas_max_abs']=float(np.max(np.abs(TS[['Hogar','Unidad médica (pública + privada)','Otro/NE']].sum(axis=1)-TS.Total)))
assert max(v['tasas_observadas_max_abs'],v['descomposicion_tasas_max_abs'])<1e-9
v['comprobaciones']['tasas2024_observadas']=True
assert len(T['pares_cambio_nivel'])==3 and len(T['pares_cambio_pendiente'])==3
assert T['covarianza_edades']['factor']==144/143
v['comprobaciones']['pares_nivel_y_pendiente_con_Holm']=True
v['comprobaciones']['correccion_edad144_143_preservada']=True
assert T['metodo_poisson']['convergencia'] and L['diagnostico_multinomial']['inferencial']
assert all(d['inferencial'] for k,d in L.items() if k.startswith(('principal_','anual_post_','anios_')))
v['comprobaciones']['principales_y_multinomial_guardias_validas']=True
v['sha256']={name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for name,p in [('C50',RAW),('WPP',WPP),('series',SERIES)]}

def diffs(a,b,path=''):
    out={}
    if isinstance(a,dict):
        for k,value in a.items():
            if k in b: out.update(diffs(value,b[k],path+'/'+k))
    elif isinstance(a,list):
        for j,(x,y) in enumerate(zip(a,b)):out.update(diffs(x,y,path+'/'+str(j)))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        out[path]=abs(float(a)-float(b))
    return out
if args.baseline:
    old=args.baseline.resolve()
    OL=json.loads((old/'resultados/lugar_defuncion.json').read_text())
    OT=json.loads((old/'resultados/tendencia_ampliaciones.json').read_text())
    df={}
    for key in L:
        if key.startswith(('principal_','anual_post_','anios_')):df.update(diffs(OL[key],L[key],key))
    v['cotejo_v10_principales_lugar']={'n_valores':len(df),'max_abs':max(df.values()),'max_campo':max(df,key=df.get)}
    assert max(df.values())<1e-8
    df=diffs(OT,T)
    v['cotejo_v10_tendencia_existente']={'n_valores':len(df),'max_abs':max(df.values()),'max_campo':max(df,key=df.get)}
    assert max(df.values())<1e-8
    v['comparacion_conocido']={s:{'n_v10':OL['sens_lugar_conocido_'+s]['n_defunciones'],
        'n_v11':L['sens_lugar_conocido_'+s]['n_defunciones'],
        'dif2023_v10':OL['sens_lugar_conocido_'+s]['periodos']['2023']['diferencia_pp'],
        'dif2023_v11':L['sens_lugar_conocido_'+s]['periodos']['2023']['diferencia_pp']} for s in ['Hogar','Unidad pública','Unidad privada']}
v['estado']='PASS'
(ROOT/'resultados/verificacion_v11.json').write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
print(json.dumps(v,indent=2,ensure_ascii=False))
