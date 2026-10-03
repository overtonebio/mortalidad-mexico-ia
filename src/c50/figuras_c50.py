"""Figuras del análisis secundario exploratorio de C50 (v11). Salida: ../figuras/*.pdf|png|svg.
Archivo interno Figura6 (figura5 del manuscrito v11): tendencia nacional y lugar de defunción. Figuras S (suplemento): variantes de tendencia,
proporciones ajustadas anuales, sensibilidades de lugar y diagnóstico de residuos."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager as fm
for f in ['Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf']:
    p = '/System/Library/Fonts/Supplemental/' + f
    if os.path.exists(p): fm.fontManager.addfont(p)
mpl.rcParams.update({'font.family': 'Arial' if os.path.exists('/System/Library/Fonts/Supplemental/Arial.ttf') else 'DejaVu Sans',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300})
COL = {'Hogar': '#1f4e79', 'Unidad pública': '#c55a11', 'Unidad privada': '#2a9d8f', 'Otro/NE': '#7f7f7f'}
ETQ = {'Hogar': 'Hogar', 'Unidad pública': 'Unidad médica pública', 'Unidad privada': 'Unidad médica privada', 'Otro/NE': 'Otro lugar o no especificado'}
CUT = pd.Timestamp('2020-03-01'); RES = '../resultados/'; FIG = '../figuras/'
def letra(ax, s, dx=-0.10, dy=1.04): ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')
def guardar(fig, nom):
    for ext in ['pdf', 'png', 'svg']: fig.savefig(f'{FIG}{nom}.{ext}', bbox_inches='tight', pad_inches=0.04)
    fig.canvas.draw()  # reset renderer after mixed PDF/PNG/SVG backends before diagnostics
    r = fig.canvas.get_renderer(); tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:] if ba.overlaps(bb)]
    print(nom, 'solapamientos de texto:', ov[:8])

P = pd.read_csv('../paquete_original/exploracion_previa/serie_y_ajuste_principal.csv', parse_dates=['fecha'])
T = pd.read_csv('../tablas/tablaS_lugar_proporciones_estandarizadas.csv')
w = pd.read_pickle(RES + 'c50_analitico.pkl'); b = w[w.residente & (w.anio_ocur <= 2023)]
obs = b.groupby('date').sitio.value_counts(normalize=True).unstack()
TS = pd.read_csv(RES + 'tasas_estandarizadas_por_lugar_mensual.csv', parse_dates=['fecha']).set_index('fecha')

# ---------------- Figura 6 ----------------
fig = plt.figure(figsize=(7.1, 5.6)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, width_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0, 0])
ax.plot(P.fecha, P.TEE_observada, color='#9aa5b1', lw=0.6, marker='o', ms=1.6, label='Tasa mensual registrada')
pre, post = P.fecha < CUT, P.fecha >= CUT
for m in [pre, post]:
    ax.fill_between(P.fecha[m], P.ic95_media_inf[m], P.ic95_media_sup[m], color=COL['Hogar'], alpha=0.18, lw=0)
    ax.plot(P.fecha[m], P.tendencia_geometrica_desestacionalizada[m], color=COL['Hogar'], lw=1.6)
ax.plot(P.fecha[post], P.continuacion_previa_descriptiva[post], color=COL['Hogar'], lw=1.0, ls='--')
ax.axvline(CUT, color='k', lw=0.6, ls=':')
ax.set_ylabel('Tasa por 100 000 mujeres-año'); ax.set_ylim(11.9, 20.3)
ax.text(pd.Timestamp('2020-06-01'), 20.0, 'Marzo 2020', fontsize=7, va='top')
ax.plot([], [], color=COL['Hogar'], lw=1.6, label='Tendencia ajustada (IC 95 %)'); ax.plot([], [], color=COL['Hogar'], lw=1.0, ls='--', label='Continuación de la tendencia previa')
ax.legend(loc='lower right', frameon=False, handlelength=1.8, borderaxespad=0.1, fontsize=6.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('Tasa estandarizada nacional de C50, 2012-2023', loc='left'); letra(ax, 'A', -0.17)

ax = fig.add_subplot(gs[0, 1]); A = TS.groupby(TS.index.year).mean()
for k, c, lab in [('Hogar', COL['Hogar'], 'Hogar'), ('Unidad médica (pública + privada)', COL['Unidad pública'], 'Unidad médica'), ('Otro/NE', COL['Otro/NE'], 'Otro/NE')]:
    ax.plot(A.index[A.index <= 2023], A[k][A.index <= 2023], color=c, marker='o', ms=2.5, lw=1.2)
    ax.plot(A.index[A.index >= 2023], A[k][A.index >= 2023], color=c, marker='o', ms=2.5, lw=1.0, ls=':', mfc='white')
    ax.text(2024.35, A[k].iloc[-1], lab, color=c, fontsize=7, va='center')
ax.axvline(2020, color='k', lw=0.6, ls=':'); ax.set_xlim(2011.5, 2027.4); ax.set_xticks(range(2012, 2025, 3)); ax.set_ylim(0, 13)
ax.set_ylabel('Tasa por 100 000 mujeres-año'); ax.set_title('Tasa estandarizada por lugar de defunción', loc='left'); letra(ax, 'B', -0.2)
ax.text(0.02, 0.97, '2024: observado, sin corrección', transform=ax.transAxes, fontsize=6.2, va='top')

ax = fig.add_subplot(gs[1, 0])
for k in ['Hogar', 'Unidad pública']:
    M = pd.read_csv(RES + f'lugar_mensual_principal_{k.replace(" ", "_")}.csv', parse_dates=['fecha'])
    ax.scatter(obs.index, 100 * obs[k], s=2, color=COL[k], alpha=0.35, lw=0)
    for m in [M.fecha < CUT, M.fecha >= CUT]:
        ax.fill_between(M.fecha[m], 100 * M.aj_inf[m], 100 * M.aj_sup[m], color=COL[k], alpha=0.2, lw=0); ax.plot(M.fecha[m], 100 * M.ajustado[m], color=COL[k], lw=1.2)
    mm = M.fecha >= CUT; ax.plot(M.fecha[mm], 100 * M.contrafactual[mm], color=COL[k], lw=0.9, ls='--')
ax.text(pd.Timestamp('2012-03-01'), 60.5, 'Hogar', color=COL['Hogar'], fontsize=7); ax.text(pd.Timestamp('2012-03-01'), 25.0, 'Unidad médica pública', color=COL['Unidad pública'], fontsize=7)
ax.axvline(CUT, color='k', lw=0.6, ls=':'); ax.set_ylim(15, 82); ax.set_ylabel('Defunciones por C50 (%)')
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('Proporción ajustada por edad, entidad y mes', loc='left'); letra(ax, 'C', -0.17)
ax.plot([], [], color='k', lw=1.2, label='Ajustada (IC 95 %)'); ax.plot([], [], color='k', lw=0.9, ls='--', label='Continuación previa'); ax.scatter([], [], s=6, color='k', alpha=0.4, label='Observada'); ax.set_yticks(range(20, 80, 10))
ax.legend(loc='upper center', frameon=False, ncol=3, handlelength=1.4, borderaxespad=0.1, bbox_to_anchor=(0.53, 1.0), fontsize=6.5, columnspacing=0.7, handletextpad=0.4)

ax = fig.add_subplot(gs[1, 1]); per = ['mar-dic 2020', '2021', '2022', '2023']; off = {'Hogar': -0.27, 'Unidad pública': -0.09, 'Unidad privada': 0.09, 'Otro/NE': 0.27}
for k in COL:
    d = T[(T.analisis == 'principal') & (T.desenlace == k)].set_index('periodo').loc[per]
    y = np.arange(len(per)) + off[k]
    ax.errorbar(d.diferencia_pp, y, xerr=[d.diferencia_pp - d.dif_ic_inf, d.dif_ic_sup - d.diferencia_pp], fmt='o', ms=3, color=COL[k], elinewidth=0.9, capsize=0, label=ETQ[k])
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(range(len(per))); ax.set_yticklabels(['Mar-dic 2020', '2021', '2022', '2023']); ax.invert_yaxis()
ax.set_xlabel('Diferencia frente a la continuación previa (pp)'); ax.set_title('Diferencia ajustada por periodo (IC 95 %)', loc='left'); letra(ax, 'D', -0.2)
ax.set_ylim(5.0, -0.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=COL[k], marker='o', ms=3.5, lw=0) for k in COL], labels=['Hogar', 'Unidad pública', 'Unidad privada', 'Otro/NE'], loc='lower center', bbox_to_anchor=(0.56, 0.0), frameon=False, fontsize=6.5, ncol=2, handletextpad=0.2, borderaxespad=0.1, columnspacing=0.8); ax.set_xlim(-16, 14)
guardar(fig, 'Figura6_C50_tendencia_y_lugar'); plt.close(fig)

# ---------------- Figura S: variantes de tendencia ----------------
E = pd.read_csv('../paquete_original/exploracion_previa/estimaciones_ITS.csv')
Aj = pd.read_csv('../tablas/tablaS_tendencia_ampliaciones.csv')
filas = [('Principal (HAC 12)', E[(E.analisis == 'principal')]), ('HAC 6 meses', E[E.analisis == 'HAC6']), ('Excluye 2020 (corte ene 2021)', E[E.analisis == 'sin_2020']),
         ('Incluye 2024 corregido', E[E.analisis == 'incluye_2024_corregido']), ('Solo residentes 1-32', Aj[Aj.analisis == 'solo residentes 1-32']),
         ('2023 + registro tardío (máx.)', Aj[Aj.analisis == '2023 + registro tardío (máximo)']), ('Poisson robusto por edad', Aj[Aj.analisis == 'cuasi-Poisson por edad']),
         ('20-39 años (modelo conjunto)', Aj[(Aj.analisis == 'edad, modelo conjunto DK') & (Aj.serie == '20-39')]),
         ('40-59 años (modelo conjunto)', Aj[(Aj.analisis == 'edad, modelo conjunto DK') & (Aj.serie == '40-59')]),
         ('60 años o más (modelo conjunto)', Aj[(Aj.analisis == 'edad, modelo conjunto DK') & (Aj.serie == '60+')])]
fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.6), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, par, tit in zip(axs, ['pendiente_pre', 'pendiente_post', 'salto_nivel'], ['Pendiente previa (%/año)', 'Pendiente posterior (%/año)', 'Cambio de nivel en el corte (%)']):
    for i, (lab, d) in enumerate(filas):
        r = d[d.parametro == par].iloc[0]; c = COL['Hogar'] if i < 7 else '#555555'
        ax.errorbar(r.estimacion_pct, i, xerr=[[r.estimacion_pct - r.ic95_inf], [r.ic95_sup - r.estimacion_pct]], fmt='o' if i < 7 else 's', ms=3, color=c, elinewidth=0.9)
    ax.axvline(0, color='k', lw=0.6); ax.set_title(tit, loc='left'); ax.axhline(6.5, color='#bbbbbb', lw=0.5)
axs[0].set_yticks(range(len(filas))); axs[0].set_yticklabels([f[0] for f in filas]); axs[0].invert_yaxis()
for a, l in zip(axs, 'ABC'): letra(a, l, -0.08 if l != 'A' else -0.10)
guardar(fig, 'FiguraS_C50_variantes_tendencia'); plt.close(fig)

# ---------------- Figura S: proporciones anuales ajustadas y sensibilidades de lugar ----------------
An = pd.read_csv('../tablas/tablaS_lugar_proporcion_ajustada_anual.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.6), gridspec_kw={'wspace': 0.75, 'width_ratios': [1, 1.15]})
ax = axs[0]
for k in COL:
    d = An[An.desenlace == k]; ax.errorbar(d.anio, d.ajustado_pct, yerr=[d.ajustado_pct - d.ic95_inf, d.ic95_sup - d.ajustado_pct], color=COL[k], marker='o', ms=2.5, lw=1.1, elinewidth=0.7, label=ETQ[k])
ax.axvline(2020, color='k', lw=0.6, ls=':'); ax.set_ylim(0, 72); ax.set_ylabel('Defunciones por C50 (%)'); ax.set_xticks(range(2012, 2024, 2))
ax.set_title('Proporción anual ajustada', loc='left'); ax.legend(frameon=False, fontsize=6.5, loc='center right', bbox_to_anchor=(1.0, 0.62)); letra(ax, 'A', -0.16)
ax = axs[1]
S = [('principal', 'Principal'), ('anual_post', 'Indicadores anuales'), ('sens_abril2020', 'Corte abril 2020'), ('sens_HAC6', 'HAC 6 meses'), ('sens_lugar_conocido', 'Solo lugar conocido'),
     ('sens_region', 'Región en vez de entidad'), ('sens_no_residentes', 'Incluye no residentes'), ('sens_2024_ponderado_completitud', 'Incluye 2024 ponderado')]
for j, (k, of) in enumerate([('Hogar', -0.2), ('Unidad pública', 0.0), ('Unidad privada', 0.2)]):
    for i, (a, lab) in enumerate(S):
        r = T[(T.analisis == a) & (T.desenlace == k) & (T.periodo == '2023')].iloc[0]
        ax.errorbar(r.diferencia_pp, i + of, xerr=[[r.diferencia_pp - r.dif_ic_inf], [r.dif_ic_sup - r.diferencia_pp]], fmt='o', ms=2.8, color=COL[k], elinewidth=0.8, label=ETQ[k] if i == 0 else None)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(range(len(S))); ax.set_yticklabels([s[1] for s in S]); ax.invert_yaxis()
ax.set_xlabel('Diferencia en 2023 frente a la continuación previa (pp)'); ax.set_title('Sensibilidades, 2023', loc='left'); ax.legend(frameon=False, fontsize=6.5, loc='upper center', ncol=3, bbox_to_anchor=(0.2, -0.17), columnspacing=0.8, handletextpad=0.2); letra(ax, 'B', -0.62)
guardar(fig, 'FiguraS_C50_lugar_anual_y_sensibilidades'); plt.close(fig)

# ---------------- Figura S: diagnóstico de residuos ----------------
Rz = pd.read_csv(RES + 'residuos_ITS_principal.csv', parse_dates=['fecha']); D = json.load(open(RES + 'tendencia_ampliaciones.json'))['diagnostico_principal']
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.6, 1]})
axs[0].plot(Rz.fecha, Rz.residuo_log, color=COL['Hogar'], lw=0.7, marker='o', ms=1.5); axs[0].axhline(0, color='k', lw=0.6); axs[0].axvline(CUT, color='k', lw=0.6, ls=':')
axs[0].set_ylabel('Residuo (escala log)'); axs[0].set_title('Residuos del modelo principal, 2012-2023', loc='left'); letra(axs[0], 'A', -0.12)
ac = D['acf_1_12']; axs[1].vlines(range(1, 13), 0, ac, color=COL['Hogar'], lw=1.2); axs[1].plot(range(1, 13), ac, 'o', color=COL['Hogar'], ms=2.5)
lim = 1.96 / np.sqrt(len(Rz)); axs[1].axhspan(-lim, lim, color='#dddddd', lw=0); axs[1].axhline(0, color='k', lw=0.6); axs[1].set_ylim(-0.35, 0.35)
axs[1].set_xticks(range(1, 13)); axs[1].set_xlabel('Rezago (meses)'); axs[1].set_ylabel('Autocorrelación'); axs[1].set_title('Autocorrelación de residuos', loc='left'); letra(axs[1], 'B', -0.25)
axs[1].text(12.3, 0.30, f"Ljung-Box p = {D['ljung_box']['12']['lb_pvalue']:.2f} (12)", ha='right', fontsize=6.5)
guardar(fig, 'FiguraS_C50_residuos'); plt.close(fig)
