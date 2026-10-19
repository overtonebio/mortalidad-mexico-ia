"""Figuras v9 (se ejecuta desde la raíz del paquete; entradas en entradas/, datos/ y resultados/; salidas en documentos/figuras/).
Basado en figuras v5. Reutiliza figs_v4.py (paleta, marcadores, fig1/fig3 base) y redefine:
 fig0  esquema metodológico (nueva figura 1);
 fig2b validación de la detección con umbrales emparejados (nueva figura 3);
 fig4b aplicación al cáncer de mama con intervalo entre series (nueva figura 4);
 figS1b, figS2b, figS3b (suplemento).
Se ejecuta desde v5/build con apply_figure_style() y panel_letter() cargados (skill figure-style)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import os as _os
_D = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in globals() else 'codigo/documentos'
exec(open(_os.path.join(_D, 'figs_v4.py')).read())
from estilo_figuras import apply_figure_style, panel_letter
OUT = 'documentos/figuras/'
RES = 'resultados/'
ENS = 'Ensamble (mediana)'
YT = ["'20", "'21", "'22", "'23", "'24"]

# ------------------------------------------------------------ Figura 1: esquema
def fig0(out=OUT + 'fig1_esquema_v9.png'):
    fig = plt.figure(figsize=(W, 4.6)); gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(2012, 2025.05); ax.set_ylim(-0.6, 3.7)
    rows = [('Origen dic-2014', 2015, 2018, 2019, 'Validación'), ('Origen dic-2015', 2016, 2019, 2020, 'Validación'),
            ('Origen dic-2016', 2017, 2020, None, 'Validación y placebo'), ('Origen dic-2019', 2020, 2024, 2025, 'Contrafactual')]
    for i, (lab, a, b, c, kind) in enumerate(rows):
        y = 3 - i
        ax.add_patch(Rectangle((2012, y - 0.3), a - 2012, 0.6, color='#BFD3E6', lw=0))
        col = '#F4B183' if kind != 'Contrafactual' else '#D55E00'
        ax.add_patch(Rectangle((a, y - 0.3), b - a, 0.6, color=col, lw=0, alpha=0.9 if kind == 'Contrafactual' else 0.8))
        if c: ax.add_patch(Rectangle((b, y - 0.3), c - b, 0.6, facecolor='white', edgecolor=col, hatch='////', lw=0.8))
        ax.text(2011.9, y, lab, ha='right', va='center', fontsize=7)
    ax.text(2013.5, 3, 'Entrenamiento', ha='center', va='center', fontsize=6.5)
    ax.text(2016.5, 3, 'Prueba, 36 meses', ha='center', va='center', fontsize=6.5)
    ax.text(2018.5, 3.45, '4.º año (48 meses)', ha='center', va='bottom', fontsize=6.5)
    ax.text(2016.0, 0, 'Entrenamiento: 2012-2019', ha='center', va='center', fontsize=6.5)
    ax.text(2022.0, 0, 'Esperado sin pandemia 2020-2023', ha='center', va='center', fontsize=6.5, color='white')
    ax.text(2024.5, 0.45, '2024: sin\nvalidar', ha='center', va='bottom', fontsize=6.5)
    ax.axvline(2020, color=ZERO, lw=0.7, ls=':'); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks(range(2012, 2026, 2)); T(ax, 'Datos del estudio: periodos de entrenamiento y prueba')
    panel_letter(ax, 'a', dx=-0.19)
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
    boxes = [(1, 'Datos', 'Defunciones\nINEGI 2012-2024\n2 870 series\nmensuales (causa,\nsexo, entidad, edad)', '#E8EEF7'),
             (21, 'Pronóstico', '10 modelos:\n4 clásicos,\n3 fundacionales,\n2 globales\ny ensamble', '#E8EEF7'),
             (41, 'Exactitud', 'Error del\npronóstico (WQL;\nmenor = mejor)\ny cobertura\nde intervalos', '#FBE5D6'),
             (61, 'Falsas alarmas', 'Placebo\n2017-2019 y\nexceso simulado\nde +10 % y +20 %', '#FBE5D6'),
             (81, 'Aplicación', 'Cáncer de mama:\nobservado frente\na esperado (%),\nintervalo de\npredicción de 90 %', '#F8CBAD')]
    for x, head, body, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 2), 17.5, 34, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec='0.4', lw=0.7))
        ax.text(x + 8.75, 32.5, head, ha='center', va='top', fontsize=7.5, fontweight='bold')
        ax.text(x + 8.75, 26, body, ha='center', va='top', fontsize=6.5, linespacing=1.35)
    for x in (19.2, 39.2, 59.2, 79.2):
        ax.annotate('', xy=(x + 1.6, 19), xytext=(x - 0.2, 19), arrowprops=dict(arrowstyle='-|>', color='0.3', lw=0.9))
    T(ax, 'Flujo del estudio'); panel_letter(ax, 'b', dx=-0.02)
    _save(fig, out)

# ------------------------------------------------------------ Figura 3: validación
def fig2b(plac_m, nat, out=OUT + 'fig3_validacion_exceso_v9.png'):
    R2 = pd.read_csv(RES + 'p2_deteccion_resultados.csv'); F = R2[R2.diseno.str.startswith('D1') & (R2.bloque == '1-3')].set_index('modelo')
    B = R2[R2.diseno.str.startswith('D1') & (R2.bloque != '1-3')]
    fig = plt.figure(figsize=(W, 6.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.62)
    models = ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5',
              'LightGBM global', 'N-HiTS global', ENS]
    order = F.loc[models].sort_values('det20').index.tolist()
    ax = fig.add_subplot(gs[0, 0]); ax.axvline(5, color=ZERO, lw=0.7, ls=':')
    ax.text(5.2, len(order) - 0.45, 'objetivo 5 %', fontsize=6, ha='left', va='bottom', color=ZERO)
    YM = {'2017': ('<', -0.22), '2018': ('^', 0), '2019': ('>', 0.22)}
    for i, m in enumerate(order):
        r = F.loc[m]
        ax.plot([r.fa_sup_lo, r.fa_sup_hi], [i, i], color=MC[m], lw=1.1)
        ax.plot(r.fa_sup, i, marker=MK[m], color=MC[m], ms=5, ls='none', zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT.get(m, m) for m in order]); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel('Falsas alarmas superiores en 2017-2019 (%)'); ax.set_xlim(0, 13)
    T(ax, 'Falsas alarmas unilaterales'); panel_letter(ax, 'a', dx=-0.45)
    ax = fig.add_subplot(gs[0, 1])
    for i, m in enumerate(order):
        r = F.loc[m]
        ax.plot([r.det20_lo, r.det20_hi], [i, i], color=MC[m], lw=1.1, alpha=0.9)
        ax.plot(r.det20, i, marker=MK[m], color=MC[m], ms=5, ls='none', zorder=3)
        ax.plot(r.det10, i, marker=MK[m], mfc='white', mec=MC[m], ms=4.5, ls='none', zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([]); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel('Series-año con el exceso detectado (%)'); ax.set_xlim(0, 64)
    h = [Line2D([], [], marker='o', color='0.3', ls='none', ms=4.5), Line2D([], [], marker='o', mfc='white', mec='0.3', ls='none', ms=4.5)]
    ax.legend(h, ['exceso +20 % (IC 95 %)', 'exceso +10 %'], frameon=False, loc='lower right', fontsize=6, handletextpad=0.2)
    T(ax, 'Detección con el mismo umbral'); panel_letter(ax, 'b', dx=-0.08)
    ax = fig.add_subplot(gs[1, :])
    a = nat[(nat.tipo == 'Cánceres') & (nat.model == ENS) & (nat.year == 2022)].copy()
    a['site'] = a.grp.str.replace('C. ', '', regex=False).str.capitalize().str.replace('snc', 'SNC')
    ordr = a.groupby('site').pscore.mean().sort_values().index.tolist()
    SX = {1: ('Hombres', '#1F4E9C', 's', 0.17), 2: ('Mujeres', '#B2182B', 'o', -0.17)}
    for i, st in enumerate(ordr):
        for sx, (nm_, c, mk, off) in SX.items():
            r = a[(a.site == st) & (a.sex == sx)]
            if len(r):
                r = r.iloc[0]; ax.errorbar(i + off, r.pscore, yerr=[[r.pscore - r.lo], [r.hi - r.pscore]], fmt=mk, color=c, ms=4, capsize=1.8, lw=0.9, zorder=3)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks(range(len(ordr))); ax.set_xticklabels(ordr, rotation=35, ha='right', rotation_mode='anchor')
    for tl in ax.get_xticklabels():
        if tl.get_text() in ('Mama', 'Leucemias', 'Linfomas y mieloma'): tl.set_fontweight('bold')
    ax.set_xlim(-0.6, len(ordr) - 0.4); ax.set_ylabel('Diferencia 2022 vs esperado (%)')
    nd = int((a.pscore < 0).sum())
    ax.legend([Line2D([], [], color=c, marker=mk, ls='none', ms=4) for _, (_, c, mk, _) in SX.items()], [v[0] for v in SX.values()], loc='upper right', frameon=False, ncol=2)
    T(ax, f'Cánceres, 2022: {nd} de {len(a)} series nacionales con estimación negativa; ninguna distinguible de cero')
    panel_letter(ax, 'c', dx=-0.09)
    _save(fig, out)

# ------------------------------------------------------------ Figura 4: cáncer de mama
def fig4b(out=OUT + 'fig4_contrafactual_TEE_C50_v9.png'):
    fc = pd.read_csv('entradas/tsfm_contrafactual_mensual.csv', parse_dates=['fecha']); fc['model'] = fc.model.replace(REN)
    Sx = pd.read_csv('datos/series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
    bm = pd.read_csv('entradas/tsfm_backtest_mensual.csv'); bm['model'] = bm.model.replace(REN)
    ex2 = pd.read_csv('entradas/tsfm_exceso_anual.csv'); ex2['model'] = ex2.model.replace(REN)
    O = pd.read_csv(RES + 'p1_intervalos_mama_entre_series.csv')
    order = ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5', ENS]
    fig = plt.figure(figsize=(W, 5.7)); gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.15], hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(gs[0, :]); g = fc[(fc.target == 'asmr') & (fc.model == ENS)]
    g23 = g[g.fecha < pd.Timestamp('2024-01-01')]; g24 = g[g.fecha >= pd.Timestamp('2023-12-01')]
    ax.fill_between(g23.fecha, g23.q10, g23.q90, color='#F4B183', alpha=0.55, lw=0, label='Intervalo de predicción de 80 % (2020-2023)')
    ax.plot(Sx.index, Sx.asmr, color='black', lw=0.9, label='Observada')
    ax.plot(g23.fecha, g23.pred, color='#D55E00', lw=1.6, label='Esperada sin pandemia (ensamble)')
    ax.plot(g24.fecha, g24.pred, color='#D55E00', lw=1.2, ls='--')
    ax.axvline(pd.Timestamp('2020-01-01'), color=ZERO, lw=0.7, ls=':')
    ax.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'), color=SHADE, lw=0, zorder=0)
    ax.set_ylabel('TEE por 100 000\nmujeres-año'); ax.set_ylim(13.5, 23.8); ax.margins(x=0.01)
    ax.legend(frameon=False, loc='upper left', ncol=3)
    T(ax, 'Tasa mensual estandarizada por edad, mujeres ≥ 20 años'); panel_letter(ax, 'a', dx=-0.07)
    ax = fig.add_subplot(gs[1, 0]); b = bm[bm.target == 'asmr']; rng = np.random.default_rng(3)
    for i, m in enumerate(order):
        v = b[b.model == m].MASE.values
        ax.plot(v, np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)), marker=MK[m], color=MC[m], ms=2.5, ls='none', alpha=0.55)
        ax.plot([v.mean()] * 2, [i - 0.32, i + 0.32], color=MC[m], lw=2.2)
    ax.axvline(1, color=ZERO, lw=0.6, ls=':'); ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT.get(m, m) for m in order])
    ax.invert_yaxis(); ax.set_xlabel('MASE (menor = mejor)'); ax.set_xlim(0.5, 2.1)
    n_out = int((b[b.model.isin(order)].MASE > 2.1).sum())
    if n_out: ax.text(2.08, 2.5, f'{n_out} valores > 2.1 →', fontsize=5.5, ha='right', va='center', color=MC['SARIMA'])
    T(ax, 'Validación 2015-2019'); panel_letter(ax, 'b', dx=-0.62)
    ax = fig.add_subplot(gs[1, 1]); e = ex2[ex2.target == 'asmr']; o = O[O.target == 'asmr'].set_index('year')
    ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for j, m in enumerate(order[:-1]):
        gg = e[e.model == m]; ax.plot(gg.year + (j - 3) * 0.075 - 0.1, gg.pscore, marker=MK[m], color=MC[m], ms=2.8, ls='none', zorder=3)
    yy = np.arange(2020, 2024)
    ax.errorbar(yy + 0.3, o.loc[yy, 'pscore'], yerr=[o.loc[yy, 'pscore'] - o.loc[yy, 'lo_todas'], o.loc[yy, 'hi_todas'] - o.loc[yy, 'pscore']],
                fmt='none', ecolor='black', capsize=2, lw=0.9, zorder=4)
    ax.plot(np.arange(2020, 2025) + 0.3, o.loc[2020:2024, 'pscore'], 'D', color='black', ms=3.5, zorder=5)
    ax.text(2024.0, -18.0, 'sin\nintervalo\nvalidado', ha='center', va='top', fontsize=5.5, color='0.25')
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_xticks(range(2020, 2025))
    ax.set_xticklabels(YT); ax.set_ylim(-28, 17)
    T(ax, 'Por modelo'); panel_letter(ax, 'c', dx=-0.3)
    ax = fig.add_subplot(gs[1, 2]); AC = {'asmr_20-39': ('20-39', '#3B8BC2', 'o'), 'asmr_40-59': ('40-59', '#1F4E9C', 's'), 'asmr_60+': ('≥ 60', '#0B1F3F', 'D')}
    ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (tg, (lb, cc, mk)) in enumerate(AC.items()):
        gg = O[O.target == tg].set_index('year'); x = np.arange(2020, 2024) + (k - 1) * 0.22
        ax.errorbar(x, gg.loc[2020:2023, 'pscore'], yerr=[gg.loc[2020:2023, 'pscore'] - gg.loc[2020:2023, 'lo_todas'], gg.loc[2020:2023, 'hi_todas'] - gg.loc[2020:2023, 'pscore']],
                    fmt=mk, color=cc, ms=3, capsize=1.5, lw=0.8, label=lb + ' años')
        ax.plot(2024 + (k - 1) * 0.22, gg.loc[2024, 'pscore'], marker=mk, color=cc, ms=3, ls='none')
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(YT)
    ax.set_ylim(-40, 56); ax.set_ylabel('Diferencia vs esperado (%)')
    ax.legend(frameon=False, loc='upper left', fontsize=6, handletextpad=0.2, borderaxespad=0.1, ncol=3, columnspacing=0.6)
    T(ax, 'Por edad (ensamble)'); panel_letter(ax, 'd', dx=-0.3)
    _family_legend(fig, order, y=-0.035, ncol=8)
    _save(fig, out)

# ------------------------------------------------------------ Figura S1
def figS1b(out=OUT + 'figS1_subgrupos_cosecha_v9.png'):
    O = pd.read_csv(RES + 'p1_intervalos_mama_entre_series.csv'); C3 = json.load(open(RES + 'p3_cosecha.json'))
    E = pd.read_csv(RES + 'p3_escenarios_cosecha.csv'); pl = pd.read_csv('entradas/lugar_defuncion_C50.csv', index_col=0)
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.6), gridspec_kw=dict(hspace=0.5, wspace=0.35))
    def eb(ax, tg, off, cc, mk, lab):
        g = O[O.target == tg].set_index('year'); yy = np.arange(2020, 2024)
        ax.errorbar(yy + off, g.loc[yy, 'pscore'], yerr=[g.loc[yy, 'pscore'] - g.loc[yy, 'lo_todas'], g.loc[yy, 'hi_todas'] - g.loc[yy, 'pscore']], fmt=mk, ms=3, color=cc, capsize=1.5, lw=0.8, label=lab)
        ax.plot(2024 + off, g.loc[2024, 'pscore'], marker=mk, ms=3, color=cc, ls='none')
    RC = {'Norte': ('#1B9E77', 'o'), 'Centro-Occidente': ('#D95F02', 's'), 'Centro': ('#7570B3', '^'), 'Sur-Sureste': ('#C2185B', 'D')}
    ax = axs[0, 0]; ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (r, (cc, mk)) in enumerate(RC.items()): eb(ax, f'asmr_{r}', (k - 1.5) * 0.17, cc, mk, r)
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(YT); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-45, 45)
    ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=6, handletextpad=0.2, columnspacing=0.8)
    T(ax, 'Por región (TEE)'); panel_letter(ax, 'a')
    ax = axs[0, 1]; ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (tg, (lab, cc, mk)) in enumerate({'deaths_SS': ('Con seguridad social', '#1F4E9C', 's'), 'deaths_noSS': ('Sin seguridad social', '#A6611A', 'o')}.items()):
        eb(ax, tg, (k - 0.5) * 0.24, cc, mk, lab)
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(YT); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-45, 45)
    ax.legend(frameon=False, loc='upper left', fontsize=6)
    T(ax, 'Por derechohabiencia (conteos)'); panel_letter(ax, 'b')
    ax = axs[1, 0]; dp = C3['deficit_conteos_2021_2023']; dt = C3['deficit_TEE_2021_2023']
    RRs = sorted(E.RR.unique()); xx = np.arange(len(RRs))
    for k, (f4, cc) in enumerate([(0.25, '#B39DDB'), (0.60, '#5E35B1')]):
        ref = E[(E.variante == 'referencia') & (E.f_2021_2024 == f4)].set_index('RR').loc[RRs]
        rng_ = E[E.f_2021_2024 == f4].groupby('RR').A_2021_2023.agg(['min', 'max']).loc[RRs]
        ax.bar(xx + (k - 0.5) * 0.36, ref.A_2021_2023, width=0.34, color=cc, lw=0, label=f'f = {f4*100:.0f} % (2021-2024)')
        ax.errorbar(xx + (k - 0.5) * 0.36, ref.A_2021_2023, yerr=[ref.A_2021_2023 - rng_['min'], rng_['max'] - ref.A_2021_2023], fmt='none', ecolor='0.25', lw=0.7, capsize=1.5)
    ax.axhline(dp, color='black', lw=1.1); ax.axhline(dt, color='black', lw=0.8, ls='--')
    ax.text(-0.45, dp + 45, f'Déficit puntual: {fmt(int(round(dp)))}\n(discontinua: desde la TEE)', fontsize=6.0, va='bottom')
    ax.set_xticks(xx); ax.set_xticklabels([f'{r:.1f}' for r in RRs]); ax.set_xlabel('Riesgo relativo supuesto (escenario)')
    ax.set_ylabel('Defunciones hipotéticas\n(ventana de 3 años)'); ax.set_ylim(0, 2400)
    ax.legend(frameon=False, loc='upper left', fontsize=6, ncol=2, bbox_to_anchor=(0, 1.0))
    T(ax, 'Escenarios de riesgo competitivo'); panel_letter(ax, 'c', dx=-0.2)
    ax = axs[1, 1]; pl2 = pl.loc[2012:2024]; ax.axvspan(2019.5, 2021.5, color=SHADE, lw=0)
    ax.plot(pl2.index, pl2['Hogar'], color='#B2182B', marker='o', ms=3, lw=1.3)
    ax.plot(pl2.index, pl2['Unidad pública'], color='#1F4E9C', marker='s', ms=3, lw=1.3)
    ax.text(2012, pl2['Hogar'].iloc[0] + 3, 'Hogar', color='#B2182B', fontsize=7)
    ax.text(2012, pl2['Unidad pública'].iloc[0] + 3, 'Unidad médica pública', color='#1F4E9C', fontsize=7)
    ax.set_ylabel('Defunciones (%)'); ax.set_ylim(15, 72); ax.set_xticks(range(2012, 2025, 2))
    T(ax, 'Lugar de defunción'); panel_letter(ax, 'd')
    _save(fig, out)

# ------------------------------------------------------------ Figura S2
def figS2b(out=OUT + 'figS2_subregistro_v9.png'):
    C3 = json.load(open(RES + 'p3_cosecha.json'))
    lab = {'age': 'Edad', 'esc_g': 'Escolaridad', 'sitio_g': 'Lugar de defunción', 'ent_g': 'Entidad', 'tloc_g': 'Tamaño de localidad',
           'asist_g': 'Asistencia médica', 'civ_g': 'Estado civil', 'dh_g': 'Derechohabiencia', 'm': 'Mes'}
    cal = pd.read_csv('entradas/subregistro_calibracion.csv'); perf = pd.read_csv('entradas/subregistro_desempeno.csv').iloc[0]
    sh = pd.read_csv('entradas/subregistro_shap.csv', index_col=0).iloc[:, 0].sort_values()
    tot = pd.read_csv('entradas/subregistro_resultados.csv', index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 0.9], wspace=0.95))
    ax = axs[0]; mx = float(max(cal.pred.max(), cal.obs.max())) * 1.1
    ax.plot([0, mx], [0, mx], color='0.55', lw=0.8, ls='--'); ax.plot(cal.pred, cal.obs, 'o-', color='#5E35B1', ms=3.5, lw=1.2)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Proporción observada')
    ax.text(0.04, 0.96, f"AUC {perf.AUC:.2f}\n(solo edad {perf.AUC_edad:.2f})", transform=ax.transAxes, va='top', fontsize=6.5)
    T(ax, 'Calibración (2019)'); panel_letter(ax, 'a', dx=-0.5)
    ax = axs[1]; ax.barh(range(len(sh)), sh.values, color='#5E35B1', height=0.6)
    ax.set_yticks(range(len(sh))); ax.set_yticklabels([lab.get(i, i) for i in sh.index]); ax.set_xlabel('|SHAP| medio (log-odds)')
    T(ax, 'Importancia de variables'); panel_letter(ax, 'b', dx=-0.5)
    ax = axs[2]; ex_ = tot.loc[2020:2024, 'exceso_vs_2017_19']
    ax.bar(ex_.index, ex_.values, color=['#5E35B1'] * 4 + ['#B39DDB'], width=0.6)
    ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(YT); ax.set_ylim(0, 42); ax.set_ylabel('Defunciones estimadas/año\nvs 2017-2019')
    dp = C3['deficit_conteos_2021_2023']; x3 = C3['c50_oculto_exceso_vs_2017_19_2021_2023']
    ax.text(0.58, 0.93, f'2021-2023:\n{x3:.0f} adicionales\nestimadas', transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    T(ax, 'Cáncer de mama en\ncausas mal definidas'); panel_letter(ax, 'c', dx=-0.5)
    _save(fig, out)

# ------------------------------------------------------------ Figura S3 (histórica)
def figS3b(out=OUT + 'figS3_piloto_historico_v9.png'):
    FC = {'Chronos-2 zero-shot': '#D55E00', 'Chronos-2 + contexto entre series': '#A6611A', 'Chronos-2 LoRA (México)': '#67001F',
          'Chronos-2 LoRA + contexto': '#C2185B', 'Log-lineal': '#1F4E9C'}
    relv = pd.read_csv('entradas/finetune_rel_por_volumen.csv', index_col=0); FA = pd.read_csv('entradas/finetune_resultados.csv.gz')
    cf = pd.read_csv('entradas/finetune_contrafactual_C50.csv')
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1, 1], wspace=0.55))
    ax = axs[0]; vb = ['5-10', '10-25', '25-100', '>100']; rv = (relv - 1) * 100
    ax.axhline(0, color=FC['Chronos-2 zero-shot'], lw=1, ls='--')
    nmS = {'Chronos-2 + contexto entre series': 'Contexto', 'Chronos-2 LoRA (México)': 'LoRA', 'Chronos-2 LoRA + contexto': 'LoRA + contexto', 'Log-lineal': 'Log-lineal'}
    mks = {'Chronos-2 + contexto entre series': '^', 'Chronos-2 LoRA (México)': 'o', 'Chronos-2 LoRA + contexto': 'D', 'Log-lineal': 's'}
    for m in nmS: ax.plot(range(4), rv.loc[vb, m].values, marker=mks[m], ms=3.5, color=FC[m], lw=1.8 if m == 'Chronos-2 LoRA (México)' else 1.2, label=nmS[m])
    ax.set_xticks(range(4)); ax.set_xticklabels(vb); ax.set_xlabel('Defunciones por mes'); ax.set_ylabel('WQL vs zero-shot (%)')
    ax.legend(frameon=False, fontsize=6, loc='upper right', handlelength=1.5)
    T(ax, 'Por volumen de la serie'); panel_letter(ax, 'a', dx=-0.35)
    ax = axs[1]; mm = ['Chronos-2 + contexto entre series', 'Chronos-2 LoRA (México)', 'Chronos-2 LoRA + contexto']
    Pw = FA.pivot_table(index=['sid', 'origin'], columns='model', values='WQL'); wr = [(Pw[m] < Pw['Chronos-2 zero-shot']).mean() * 100 for m in mm]
    for i, (v, m) in enumerate(zip(wr, mm)):
        ax.plot([50, v], [i, i], color=FC[m], lw=1.0); ax.plot(v, i, 'o', color=FC[m], ms=6); ax.text(v + 2.0, i, f'{v:.0f} %', va='center', fontsize=6.5)
    ax.axvline(50, color=ZERO, lw=0.7, ls=':'); ax.text(50.5, 2.45, 'sin cambio (50 %)', fontsize=5.5, color=ZERO, va='center')
    ax.set_yticks(range(3)); ax.set_yticklabels(['Contexto', 'LoRA', 'LoRA +\ncontexto']); ax.invert_yaxis(); ax.set_ylim(2.7, -0.5)
    ax.set_xlim(45, 80); ax.set_xlabel('Serie-origen que mejoran (%)'); T(ax, f'{fmt(len(Pw))} series-origen'); panel_letter(ax, 'b', dx=-0.45)
    ax = axs[2]; c1 = cf[cf.target == 'asmr'].pivot(index='year', columns='model', values='pscore')
    ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    ax.plot(c1.index, c1['Chronos-2 zero-shot'], 'o-', color=FC['Chronos-2 zero-shot'], ms=3, lw=1.2, label='Zero-shot')
    ax.plot(c1.index, c1['Chronos-2 LoRA (México)'], 's-', color=FC['Chronos-2 LoRA (México)'], ms=3, lw=1.2, label='LoRA')
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(YT)
    ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-12, 5)
    ax.text(2024, 4.5, 'sin validar', ha='center', va='top', fontsize=5.5, color='0.25')
    ax.legend(frameon=False, fontsize=6, loc='lower left'); T(ax, 'Cáncer de mama (TEE), solo puntual'); panel_letter(ax, 'c', dx=-0.35)
    _save(fig, out)

# ------------------------------------------------------------ Figura 5: LoRA
def fig3b(cmp, out=OUT + 'fig5_adaptacion_LoRA_v9.png'):
    L = pd.read_csv(RES + 'p4b_lora_armonizado.csv').set_index('variante')
    KS = 'LoRA 1000 estricto (sin C50 ni todas las causas)'
    var = ['Completo 1000 s0', 'LoRA 300 s0', 'LoRA 1000 s0', 'LoRA 1000 sin mama', KS]
    labs = ['Ajuste completo, 1 000 pasos', 'LoRA, 300 pasos', 'LoRA, 1 000 pasos (3 semillas)', 'LoRA, sin cáncer de mama', 'LoRA, exclusión estricta*']
    cols = ['#666666', '#D6604D', '#67001F', '#B2182B', '#E7298A']
    vals = L.loc[var, 'todas_WQL_rel_geo'].values; wv = L.loc[var, 'todas_pct_series_mejoran'].values
    seeds = L.loc[['LoRA 1000 s0', 'LoRA 1000 s1', 'LoRA 1000 s2'], 'todas_WQL_rel_geo'].values
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.4), gridspec_kw=dict(width_ratios=[1.25, 0.8, 1.05], wspace=0.35))
    ax = axs[0]; y = np.arange(len(var))
    ax.barh(y, (vals - 1) * 100, color=cols, height=0.6)
    ax.plot((seeds - 1) * 100, [2] * 3, 'o', mfc='white', mec='black', ms=3.5, mew=0.7, zorder=3)
    for m, ls in [('LightGBM global', '--'), (ENS, ':')]: ax.axvline((cmp[m] - 1) * 100, color=MC[m], ls=ls, lw=1.1)
    ax.text((cmp['LightGBM global'] - 1) * 100 + 0.05, len(var) - 0.25, 'LightGBM', fontsize=6.5, color=MC['LightGBM global'], ha='left', va='center')
    ax.text((cmp[ENS] - 1) * 100 - 0.05, len(var) - 0.25, 'Ensamble', fontsize=6.5, color='black', ha='right', va='center')
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_ylim(len(var) + 0.1, -0.6); ax.set_xlim(-3.4, 0)
    ax.set_xlabel('WQL vs Chronos-2 sin adaptar (%)'); T(ax, 'Reducción media del error'); panel_letter(ax, 'a', dx=-1.0)
    ax = axs[1]; ax.axvline(50, color=ZERO, lw=0.7, ls=':')
    for i, (v, c) in enumerate(zip(wv, cols)):
        ax.plot([50, v], [i, i], color=c, lw=1.0); ax.plot(v, i, 'o', color=c, ms=6); ax.text(v + 1.6, i, f'{v:.0f} %', va='center', fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_ylim(len(var) + 0.1, -0.6); ax.set_xlim(45, 76)
    ax.set_xlabel('% de 2 870 series'); T(ax, 'Series que mejoran'); panel_letter(ax, 'b', dx=-0.12)
    ax = axs[2]; bv = [1.0] + list(L.loc[['LoRA 1000 s0', 'LoRA 1000 sin mama', KS], 'mama_WQL_rel_geo'].values)
    bc = ['#D55E00', '#67001F', '#B2182B', '#E7298A']
    ax.bar(range(4), (np.array(bv) - 1) * 100, color=bc, width=0.6); ax.axhline(0, color=ZERO, lw=0.7)
    ax.plot(0, 0, marker='_', color=bc[0], ms=14, mew=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(['Sin\nadaptar', 'Todas', 'Sin\nmama', 'Estricta*'])
    ax.set_ylabel('WQL vs sin adaptar (%)'); ax.set_ylim(-4.6, 0.5)
    for i, v in enumerate((np.array(bv) - 1) * 100):
        if i: ax.text(i, v - 0.12, f'{v:.1f} %'.replace('-', '−'), ha='center', va='top', fontsize=6.5)
    T(ax, f"Cáncer de mama (n = {int(L.loc[KS, 'mama_n'])} series)"); panel_letter(ax, 'c', dx=-0.3)
    fig.text(0.01, -0.04, '* Sin series de cáncer de mama ni de todas las causas en el entrenamiento.', fontsize=6.5)
    _save(fig, out)


def fig1_desde_numeros(out=OUT + 'fig2_benchmark_v9.png'):
    N = json.load(open('resultados/benchmark/numeros_repro.json')); FR = json.load(open('resultados/benchmark/friedman_por_serie.json'))['por_serie_WQLmedio']
    rk = pd.Series(FR['rangos']); cd = FR['CD']
    hor = pd.DataFrame({m: [N['hor'][m][k] for k in ('h1', 'h2', 'h3')] for m in N['hor']}).T
    lv = pd.DataFrame(N['by_level']).T if isinstance(next(iter(N['by_level'].values())), dict) else pd.DataFrame(N['by_level'])
    ci = pd.DataFrame(N['rel']).T[['rel', 'lo', 'hi']]
    fig1(rk, cd, hor, lv, ci, N['cov'], N['n_series'], N['n_eval'], out=out)
