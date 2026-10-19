"""Verificación posterior del análisis; no selecciona ni reestima especificaciones."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
import interaccion_localidad as a

O=Path(__file__).resolve().parent
R=json.loads((O/'resultados_interaccion.json').read_text())
d,audit=a.load_data(a.DEFAULT_RAW)
saved=np.load(O/'ajuste_principal.npz')
b=saved['beta']; cov=saved['cov']; cols=list(saved['columns'])
model={'b':b,'cov':cov,'cols':cols,'shape':'segmentado'}
strata=pd.read_csv(O/'pesos_estandarizacion.csv')

# Unidad de observación diferente: Bernoulli individual vs binomial agrupado.
Xi=a.design(d,'segmentado').reindex(columns=cols).to_numpy()
fit=sm.Logit(d.home.to_numpy(),Xi).fit(start_params=b+np.random.default_rng(20261005).normal(0,.01,len(b)),
                                    method='newton',maxiter=30,tol=1e-10,disp=0)
coef_error=float(np.max(np.abs(fit.params-b)))
assert fit.mle_retvals['converged'] and coef_error<1e-7

# Verifica derivadas mediante diferencias finitas en tres márgenes contrastantes.
checks=[]
for loc,mi in [(0,95),(0,143),(3,143)]:
    X=a.design(strata.assign(local=loc,mi=mi),'segmentado').reindex(columns=cols).to_numpy()
    wt=strata.weight.to_numpy(); p=expit(X@b)
    analytic=((wt*p*(1-p))[:,None]*X).sum(0)
    numeric=[]; eps=1e-5
    for j in range(len(b)):
        v=np.zeros(len(b));v[j]=eps
        numeric.append((wt@expit(X@(b+v))-wt@expit(X@(b-v)))/(2*eps))
    maxerr=float(np.max(np.abs(analytic-numeric)))
    assert maxerr<1e-8
    checks.append({'localidad':a.LABELS[loc],'mes':mi,'max_error_gradiente':maxerr})

# Cuánto peso del estándar corresponde a celdas edad×entidad no observadas en cada localidad.
support=[]
for loc in range(4):
    c=d[d.local==loc].groupby(['ag','ent_resid']).size().rename('observaciones').reset_index()
    w=strata.merge(c,on=['ag','ent_resid'],how='left',validate='one_to_one').fillna({'observaciones':0})
    support.append({'localidad':a.LABELS[loc], 'estratos_edad_entidad_del_estandar':len(w),
        'estratos_sin_observaciones':int((w.observaciones==0).sum()),
        'peso_sin_observaciones_pct':float(100*w.loc[w.observaciones==0,'weight'].sum()),
        'peso_con_menos_de_5_observaciones_pct':float(100*w.loc[w.observaciones<5,'weight'].sum())})

def form(v,dec=1): return f'{v:.{dec}f}'
def ci(z,dec=1): return f"{form(z['estimacion'],dec)} ({form(z['ic95_inf'],dec)} a {form(z['ic95_sup'],dec)})"
def pform(v): return '<0.001' if v<.001 else f'{v:.3f}'
M=R['models']['principal']
pd.DataFrame(support).to_csv(O/'soporte_estandarizacion.csv',index=False)
table=[]
for loc in a.LABELS:
    row={'Localidad (habitantes)':loc,'N 2012-2023':audit['por_localidad'][a.LABELS.index(loc)]}
    for yr in ['2019','mar-dic 2020','2021','2022','2023']:
        z=next(x for x in M['periodos'] if x['localidad']==loc and x['periodo']==yr and x['medida']=='ajustado_pct')
        row[f'{yr}, % (IC95%)']=ci(z)
    table.append(row)
selected=['Brecha rural - ≥100000, 2019','Brecha rural - ≥100000, 2023',
          'Cambio 2019-2023 rural','Cambio 2019-2023 ≥100000',
          'Cambio de brecha rural - ≥100000 entre 2019 y 2023']
contrasts=[{'Contraste':z['contraste'],'Diferencia, pp (IC95%)':ci(z), 'p exploratoria':pform(z['p_exploratoria'])}
           for name in selected for z in M['contrastes'] if z['contraste']==name]
sens=[]
for key,m in R['models'].items():
    dd=next(z for z in m['contrastes'] if z['contraste']==selected[-1])
    sens.append({'Modelo':key,'N':m['diagnosticos']['n'],'p interacción global':pform(m['interaction']['p']),
                 'gl':m['interaction']['gl'],'Cambio de brecha, pp (IC95%)':ci(dd),
                 'p cambio de brecha':pform(dd['p_exploratoria'])})

footnote=('Probabilidades estimadas por regresión logística segmentada, ajustada por edad quinquenal, entidad y mes del año, '
          'con interceptos y tendencias previas propios de cada localidad e interacciones de localidad con el nivel y la pendiente desde marzo de 2020. '
          'Estandarización a la distribución conjunta de edad y entidad de las 82 377 defunciones con localidad conocida de 2012-2023; '
          'misma composición en las cuatro categorías. Cada mes tiene igual peso dentro de cada periodo. Los IC95% son de parámetros y márgenes estimados, '
          'obtenidos por método delta y covarianza HAC con núcleo Bartlett y 12 rezagos; no son intervalos de predicción. '
          'pp: puntos porcentuales. Se conservaron 867 registros con lugar de defunción no especificado en el denominador principal. '
          'El cambio de brecha compara la diferencia rural−≥100 000 en 2023 con la de 2019; no equivale a un efecto causal. '
          'Análisis exploratorio motivado por la revisión del manuscrito, sin ajuste por multiplicidad.')
editorial={'table_title':'Tabla S26. Defunciones domiciliarias por cáncer de mama según tamaño de localidad: probabilidades ajustadas, contrastes y sensibilidades.',
           'panel_A_title':'A. Probabilidades ajustadas de defunción en el hogar entre las defunciones por cáncer de mama',
           'panel_A':table,'panel_B_title':'B. Brecha rural frente a localidades de al menos 100 000 habitantes y cambio entre 2019 y 2023',
           'panel_B':contrasts,'panel_C_title':'C. Sensibilidades previstas antes del ajuste de esta ampliación exploratoria',
           'panel_C':sens,'footnote':footnote}
(O/'tabla_S26_editorial.json').write_text(json.dumps(editorial,ensure_ascii=False,indent=2)+'\n')
pd.DataFrame(table).to_csv(O/'tabla_S26A_probabilidades.csv',index=False)
pd.DataFrame(contrasts).to_csv(O/'tabla_S26B_contrastes.csv',index=False)
pd.DataFrame(sens).to_csv(O/'tabla_S26C_sensibilidades.csv',index=False)

V={'status':'Verificado para uso exploratorio con limitaciones declaradas',
   'Bernoulli_individual_convergente':bool(fit.mle_retvals['converged']),
   'max_dif_beta_individual_agrupado':coef_error,'gradientes_diferencias_finitas':checks,
   'soporte_estandarizacion':support,'reconciliacion_descriptiva':audit['reconciliacion'],
   'condicion_cov':float(np.linalg.cond(cov)),'rango_cov':int(np.linalg.matrix_rank(cov)),
   'eigen_min_cov':float(np.linalg.eigvalsh(cov).min()),
   'versions':R['metadata']['versions'],'n':audit['n'],'eventos_hogar':audit['hogar'],
   'localidad_no_clasificada_n':482,'localidad_no_clasificada_pct':100*482/82859,
   'no_especificado99_n':867,'no_especificado99_pct':100*867/82377,
   'nota':'No se hace nueva selección de variantes. La sensibilidad de la interacción global al ancho HAC y al excluir99 se conserva en el reporte.'}
(O/'verificacion_independiente.json').write_text(json.dumps(V,ensure_ascii=False,indent=2,default=a.to_json)+'\n')
print(json.dumps(V,ensure_ascii=False,indent=2))
