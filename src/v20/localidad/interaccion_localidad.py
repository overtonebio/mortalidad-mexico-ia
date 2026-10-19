"""Ampliación exploratoria registrada el 5-oct-2026: comentario 132.

Ejemplo (desde la raíz de trabajo):
PYTHONPATH=analisis_mama_exploratorio_20260930/deps /ruta/python3 \
  revision_DP_20261005/analisis_localidad/interaccion_localidad.py
No modifica los datos fuente ni el repositorio editorial. Ver PLAN_20261005.txt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
import importlib.metadata
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DEFAULT_RAW = ROOT / 'revision_github_zenodo/auditoria_Codex/ejec_figuras/trabajo/c50/paquete_original/editorial/datos/c50_raw_2012_2024.csv.gz'
LABELS = ['< 2 500', '2 500-14 999', '15 000-99 999', '≥ 100 000']
MONTHS = pd.date_range('2012-01-01', '2023-12-01', freq='MS')
CUT = 98  # marzo2020 contado desde enero2012=0
PERIODS = {'2019': (84, 95), 'mar-dic 2020': (98, 107), '2021': (108, 119),
           '2022': (120, 131), '2023': (132, 143), 'mar 2020-dic 2023': (98, 143)}
Z = stats.norm.ppf(.975)


def to_json(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(type(obj).__name__)


def load_data(raw_path):
    raw = pd.read_csv(raw_path, dtype=str)
    for col in raw.columns: raw[col] = raw[col].str.strip()
    assert raw.causa_def.str.startswith('C50').all()
    for col in ['sexo', 'edad', 'mes_ocurr', 'anio_ocur', 'ent_resid', 'tloc_resid', 'sitio_ocur']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    filters = [
        ('Registros C50 conservados', np.ones(len(raw), dtype=bool)),
        ('Mujeres', raw.sexo == 2),
        ('Edad en años cumplidos >=20', raw.edad.between(4020, 4997)),
        ('Mes de ocurrencia válido', raw.mes_ocurr.between(1, 12)),
        ('Ocurrencia 2012-2023', raw.anio_ocur.between(2012, 2023)),
        ('Residentes de México (entidades 1-32)', raw.ent_resid.between(1, 32)),
        ('Tamaño de localidad conocido (1-17)', raw.tloc_resid.between(1, 17))]
    keep = np.ones(len(raw), dtype=bool)
    flow = []
    for name, f in filters:
        previous = keep.sum(); keep &= np.asarray(f)
        flow.append({'paso': name, 'n': int(keep.sum()), 'excluidos_en_paso': int(previous - keep.sum())})
    population = raw[np.logical_and.reduce([np.asarray(f) for _, f in filters[:-1]])].copy()
    miss = population.assign(localidad_ignorada=lambda x: ~x.tloc_resid.between(1,17),
                              lugar99=lambda x: x.sitio_ocur == 99).groupby('anio_ocur').agg(
        total=('sexo','size'), localidad_ignorada=('localidad_ignorada','sum'), lugar99=('lugar99','sum'))
    miss.to_csv(OUT / 'faltantes_por_anio.csv')
    d = raw.loc[keep].copy()
    assert d.sitio_ocur.isin(list(range(1, 14))+[99]).all()
    d['ag'] = np.minimum((d.edad - 4000)//5*5, 85).astype(int)
    d['local'] = pd.cut(d.tloc_resid, [0,3,6,12,17], labels=False).astype(int)
    d['mi'] = ((d.anio_ocur - 2012)*12 + d.mes_ocurr - 1).astype(int)
    d['home'] = (d.sitio_ocur == 11).astype(int)
    d.ent_resid = d.ent_resid.astype(int)
    assert set(d.mi) == set(range(144))
    assert (d.groupby('local').home.nunique() == 2).all()
    raw_desc = d.groupby(['anio_ocur','local']).agg(n=('home','size'), hogar=('home','sum'))
    raw_desc['porcentaje'] = 100*raw_desc.hogar/raw_desc.n
    raw_desc = raw_desc.reset_index(); raw_desc['localidad'] = raw_desc.local.map(dict(enumerate(LABELS)))
    raw_desc.to_csv(OUT/'descriptivos_anuales.csv',index=False)
    pd.DataFrame(flow).to_csv(OUT/'flujo_muestra.csv',index=False)
    # Reconciliación independiente con el resultado descriptivo conservado.
    ref_path = ROOT / 'Entrega_GitHub_Zenodo_GMM_v13_revisada_Codex/github/mortalidad-mexico-ia/results/reference/c50/resultados/v12_apvp_localidad/localidad_hogar.csv'
    reconcile = {'fuente': str(ref_path), 'disponible': ref_path.exists()}
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        check = ref.merge(raw_desc, left_on=['anio','localidad'], right_on=['anio_ocur','localidad'], validate='one_to_one')
        reconcile.update({'filas':len(check), 'max_diferencia_n':float((check.n_x-check.n_y).abs().max()),
                          'max_diferencia_hogar':float((check.hogar_x-check.hogar_y).abs().max()),
                          'max_diferencia_pp':float((check.hogar_pct-check.porcentaje).abs().max())})
        assert len(check)==12 and reconcile['max_diferencia_n']==0 and reconcile['max_diferencia_hogar']==0
        assert reconcile['max_diferencia_pp'] < 1e-10
    audit = {'flujo':flow, 'n':len(d), 'hogar':int(d.home.sum()), 'lugar99':int((d.sitio_ocur==99).sum()),
             'por_localidad':d.groupby('local').size().to_dict(), 'reconciliacion':reconcile,
             'tamano_codigo':d.tloc_resid.value_counts().sort_index().to_dict(),
             'sitio_codigo':d.sitio_ocur.value_counts().sort_index().to_dict(),
             'nota_duplicados':'No se eliminan filas iguales: no hay identificador individual y las variables retenidas no identifican un registro único.'}
    return d, audit


def design(t, shape):
    mi = t.mi.to_numpy()
    ti = (mi - CUT)/12  # centrado en corte para estabilidad numérica
    post = (mi >= CUT).astype(float)
    ptime = np.maximum(ti, 0)
    vals = {'const':np.ones(len(t)), 'tiempo':ti}
    for local in range(3):
        dummy = (t.local.to_numpy()==local).astype(float)
        vals[f'local_{local}']=dummy
        vals[f'pre_local_{local}']=dummy*ti
    if shape=='segmentado':
        post_terms = {'post':post, 'tiempo_post':ptime}
    else:
        post_terms = {f'periodo_{year}':((mi>=CUT)&(2012+mi//12==year)).astype(float) for year in range(2020,2024)}
    vals.update(post_terms)
    for term,value in post_terms.items():
        for local in range(3): vals[f'{term}_local_{local}']=value*(t.local.to_numpy()==local)
    for age in range(25,90,5): vals[f'edad_{age}']=(t.ag.to_numpy()==age).astype(float)
    for ent in range(2,33): vals[f'entidad_{ent}']=(t.ent_resid.to_numpy()==ent).astype(float)
    for m in range(2,13): vals[f'mes_{m}']=(mi%12+1==m).astype(float)
    return pd.DataFrame(vals, index=t.index)


def hac_cov(scores, mi, bread, lag):
    summed = np.zeros((144,scores.shape[1])); np.add.at(summed,mi,scores)
    meat = summed.T@summed
    for ell in range(1,lag+1):
        cross = summed[ell:].T@summed[:-ell]
        meat += (1-ell/(lag+1))*(cross+cross.T)
    result = bread@meat@bread*144/143
    return (result+result.T)/2


def fit_model(d, shape):
    t = d.groupby(['mi','ag','ent_resid','local'],observed=True).agg(n=('home','size'),k=('home','sum')).reset_index()
    X = design(t,shape); x=X.to_numpy(); n=t.n.to_numpy(); k=t.k.to_numpy()
    assert np.linalg.matrix_rank(x)==len(X.columns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fit = sm.GLM(np.column_stack([k,n-k]), x, family=sm.families.Binomial()).fit(tol=1e-11,maxiter=100)
    p = np.asarray(fit.fittedvalues)
    H = x.T@(x*(n*p*(1-p))[:,None]); bread=np.linalg.inv(H)
    scores = x*(k-n*p)[:,None]
    cov = hac_cov(scores,t.mi.to_numpy(),bread,12)
    external = cov_nw_groupsum(fit, 12, t.mi.to_numpy(), use_correction=False)*144/143
    crosscheck = np.max(np.abs(cov-external))/np.max(np.abs(cov))
    resid_table = t[['mi','local']].copy()
    resid_table['obs']=k; resid_table['fit']=n*p; resid_table['var']=n*p*(1-p)
    resid_table = resid_table.groupby(['mi','local']).sum().reset_index()
    resid_table['pearson']=(resid_table.obs-resid_table.fit)/np.sqrt(resid_table['var'])
    diag = {'converged':bool(fit.converged), 'n':int(n.sum()), 'home':int(k.sum()), 'n_cells':len(t), 'months':144,
            'k':x.shape[1], 'rank':int(np.linalg.matrix_rank(x)), 'H_condition':float(np.linalg.cond(H)),
            'score_max_abs':float(np.max(np.abs(scores.sum(0)))), 'cov_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
            'cov_independent_statsmodels_relative_error':float(crosscheck), 'warnings':[str(z.message) for z in w],
            'residuals_by_local':{}}
    for loc in range(4):
        rr=resid_table[resid_table.local==loc].sort_values('mi').pearson
        diag['residuals_by_local'][LABELS[loc]]={'lag1':float(rr.autocorr(1)),
            'Ljung_Box_approximate':acorr_ljungbox(rr,lags=[6,12,24],return_df=True).to_dict('index')}
    assert fit.converged and np.isfinite(fit.params).all() and np.isfinite(cov).all()
    assert diag['H_condition']<1e10 and diag['score_max_abs']/n.sum()<1e-8
    assert diag['cov_min_eigenvalue']>-1e-10 and crosscheck<1e-8
    return dict(b=fit.params, cov=cov, cols=list(X.columns), shape=shape,
                scores=scores,mi=t.mi.to_numpy(),bread=bread,diag=diag, residuals=resid_table)


def uncertainty(value, gradient, cov, multiplier=100):
    variance=float(gradient@cov@gradient)
    assert variance>=-1e-12
    se=np.sqrt(max(variance,0))
    return {'estimacion':multiplier*value, 'ic95_inf':multiplier*(value-Z*se),
            'ic95_sup':multiplier*(value+Z*se), 'error_estandar':multiplier*se,
            'p_exploratoria':float(2*stats.norm.sf(abs(value/se))) if se>0 else None}


def global_interaction(model,cov):
    ix=[i for i,c in enumerate(model['cols']) if '_local_' in c and not c.startswith('pre_')]
    b=model['b'][ix]; C=cov[np.ix_(ix,ix)]
    chi=float(b@np.linalg.solve(C,b))
    return {'chi2':chi,'gl':len(ix),'p':float(stats.chi2.sf(chi,len(ix))),
            'escala':'logit','coeficientes':[model['cols'][i] for i in ix]}


def marginals(model,strata):
    # Derivadas analíticas de la media ponderada de expit; método delta conjunto.
    vals=np.empty((4,144,2)); grads=np.empty((4,144,2,len(model['cols'])))
    post_ix=[i for i,c in enumerate(model['cols']) if c=='post' or c=='tiempo_post' or c.startswith(('post_local_','tiempo_post_local_','periodo_'))]
    for loc in range(4):
        for mi in range(144):
            t=strata.assign(mi=mi,local=loc)
            X=design(t,model['shape']).reindex(columns=model['cols']).to_numpy()
            for cf in (0,1):
                x=X.copy()
                if cf: x[:,post_ix]=0
                p=expit(x@model['b']); wt=strata.weight.to_numpy()
                vals[loc,mi,cf]=wt@p
                grads[loc,mi,cf]=((wt*p*(1-p))[:,None]*x).sum(0)
    return vals,grads


def summarize(model,cov,strata,values=None):
    vals,grads = values if values is not None else marginals(model,strata)
    table=[]; contrasts=[]
    def period(loc,name,cf):
        a,z=PERIODS[name]; return vals[loc,a:z+1,cf].mean(0),grads[loc,a:z+1,cf].mean(0)
    for loc in range(4):
        for name in PERIODS:
            value,gradient=period(loc,name,0); base,gb=period(loc,name,1)
            for typ,v,g in [('ajustado_pct',value,gradient),('extrapolacion_previa_pct',base,gb),('desvio_pp',value-base,gradient-gb)]:
                table.append({'localidad':LABELS[loc], 'periodo':name, 'medida':typ, **uncertainty(v,g,cov)})
    def add(name,v,g): contrasts.append({'contraste':name,**uncertainty(v,g,cov)})
    for name in PERIODS:
        r,rg=period(0,name,0); u,ug=period(3,name,0)
        rc,rcg=period(0,name,1); uc,ucg=period(3,name,1)
        add('Brecha rural - ≥100000, '+name,r-u,rg-ug)
        add('Diferencia entre desvíos rural - ≥100000, '+name,(r-rc)-(u-uc),(rg-rcg)-(ug-ucg))
    r19,g19=period(0,'2019',0); u19,gu19=period(3,'2019',0)
    r23,g23=period(0,'2023',0); u23,gu23=period(3,'2023',0)
    add('Cambio 2019-2023 rural',r23-r19,g23-g19)
    add('Cambio 2019-2023 ≥100000',u23-u19,gu23-gu19)
    add('Cambio de brecha rural - ≥100000 entre 2019 y 2023',(r23-u23)-(r19-u19),(g23-gu23)-(g19-gu19))
    return {'interaction':global_interaction(model,cov),'periodos':table,'contrastes':contrasts}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,default=DEFAULT_RAW); args=ap.parse_args()
    assert (OUT/'PLAN_20261005.txt').exists()
    d,audit=load_data(args.raw)
    strata=d.groupby(['ag','ent_resid']).size().rename('n').reset_index()
    strata['weight']=strata.n/strata.n.sum(); strata.to_csv(OUT/'pesos_estandarizacion.csv',index=False)
    print('N analítico:',len(d),'; hogar:',d.home.sum(),'; lugar99:',(d.sitio_ocur==99).sum(),flush=True)
    results={}
    principal=fit_model(d,'segmentado'); values=marginals(principal,strata)
    for lag in (12,6,18):
        name='principal' if lag==12 else f'sens_HAC{lag}'
        cov=hac_cov(principal['scores'],principal['mi'],principal['bread'],lag)
        results[name]=summarize(principal,cov,strata,values)
        results[name]['diagnosticos']=principal['diag']; results[name]['rezagos_HAC']=lag
        print(name,results[name]['interaction'],flush=True)
    for name,data,shape in [('sens_sin99',d[d.sitio_ocur!=99],'segmentado'),('sens_desvios_anuales',d,'anual')]:
        model=fit_model(data,shape)
        results[name]=summarize(model,model['cov'],strata)
        results[name]['diagnosticos']=model['diag'];results[name]['rezagos_HAC']=12
        np.savez_compressed(OUT/f'ajuste_{name}.npz',beta=model['b'],cov=model['cov'],columns=model['cols'])
        print(name,results[name]['interaction'],flush=True)
    np.savez_compressed(OUT/'ajuste_principal.npz',beta=principal['b'],cov=principal['cov'],columns=principal['cols'])
    principal['residuals'].to_csv(OUT/'residuos_mensuales_localidad.csv',index=False)
    rows=[]
    for loc in range(4):
        for mi,date in enumerate(MONTHS):
            for cf,label in [(0,'ajustado_pct'),(1,'extrapolacion_previa_pct')]:
                rows.append({'localidad':LABELS[loc],'fecha':str(date.date()),'medida':label,
                             **uncertainty(values[0][loc,mi,cf],values[1][loc,mi,cf],principal['cov'])})
    pd.DataFrame(rows).to_csv(OUT/'ajuste_mensual_principal.csv',index=False)
    for key,item in results.items():
        pd.DataFrame(item['periodos']).to_csv(OUT/f'probabilidades_{key}.csv',index=False)
        pd.DataFrame(item['contrastes']).to_csv(OUT/f'contrastes_{key}.csv',index=False)
    metadata={'fecha_plan':'2026-10-05 America/Mexico_City','source':str(args.raw),'source_sha256':hashlib.sha256(args.raw.read_bytes()).hexdigest(),
              'plan_sha256':hashlib.sha256((OUT/'PLAN_20261005.txt').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'python':sys.version,
              'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels']},
              'inferencia':'Exploratoria. IC95% delta con covarianza HAC de scores mensuales. p sin ajuste por multiplicidad.',
              'standard':'Distribución conjunta edad×entidad de la muestra principal con localidad conocida2012-2023; idéntica en los cuatro grupos y sensibilidades.',
              'month_weight':'Igual peso de cada mes dentro de cada periodo.',
              'interpretacion':'Proporciones entre defunciones por C50; no tasas de mortalidad ni efectos causales. La extrapolación representa continuación modelada de tendencias previas propias de cada localidad.'}
    result={'metadata':metadata,'audit':audit,'models':results}
    (OUT/'resultados_interaccion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=to_json,allow_nan=False)+'\n')
    print('Finalizado; resultados_interaccion.json',flush=True)


if __name__=='__main__': main()
