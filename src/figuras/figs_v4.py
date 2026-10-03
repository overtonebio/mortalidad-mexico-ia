"""Figuras v4 (auditoría: rangos por serie, intervalos 2023 validados a 48 meses, semiancho mediano en fig. 2a).
Figuras v3: proporciones equilibradas, títulos centrados y paleta con contraste >= 3:1 sobre blanco (WCAG 2.1, objetos gráficos).
Requiere apply_figure_style() y panel_letter() (skill figure-style) cargados en el entorno."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fmt = lambda x: f'{x:,}'.replace(',', ' ')
MC = {'Ingenuo estacional': '#666666', 'Log-lineal': '#1F4E9C', 'SARIMA': '#00838F', 'ETS amortiguado': '#2E7D32',
      'Chronos-2': '#D55E00', 'Chronos-Bolt': '#9C6B00', 'TimesFM-2.5': '#C2185B',
      'LightGBM global': '#5E35B1', 'N-HiTS global': '#795548', 'Ensamble (mediana)': '#000000'}
MK = {'Ingenuo estacional': 's', 'Log-lineal': 's', 'SARIMA': 's', 'ETS amortiguado': 's',
      'Chronos-2': 'o', 'Chronos-Bolt': 'o', 'TimesFM-2.5': 'o', 'LightGBM global': '^', 'N-HiTS global': '^', 'Ensamble (mediana)': 'D'}
SHORT = {'LightGBM global': 'LightGBM', 'N-HiTS global': 'N-HiTS', 'Ensamble (mediana)': 'Ensamble', 'Ingenuo estacional': 'Ingenuo',
         'ETS amortiguado': 'ETS', 'TimesFM-2.5': 'TimesFM-2.5'}
REN = {'Naive estacional': 'Ingenuo estacional', 'Log-lineal (tendencia+mes)': 'Log-lineal'}
GRID = '0.90'; ZERO = '0.35'; SHADE = '0.93'
W = 7.2  # ancho de doble columna (pulgadas)

def T(ax, t):
    ax.set_title(t, loc='center', pad=7)

def _save(fig, out):
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05); plt.close(fig)

def _family_legend(fig, models, y=0.0, ncol=5):
    h = [Line2D([], [], color=MC[m], marker=MK[m], ms=4, lw=1.3) for m in models]
    fig.legend(h, [SHORT.get(m, m) for m in models], loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               handlelength=1.8, columnspacing=1.2)

# ---------------------------------------------------------------- Figura 1
def fig1(rk_all, cd_all, hor, by_level, ci, cov, NS, N_all, out='fig_benchmark_v2.png'):
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.9), gridspec_kw=dict(hspace=0.55, wspace=0.42))
    fig.subplots_adjust(bottom=0.13)
    # a: rango promedio
    ax = axs[0, 0]; r = rk_all.sort_values()
    for i, (m, v) in enumerate(r.items()):
        ax.plot([3, v], [i, i], color=GRID, lw=0.8, zorder=1)
        ax.plot(v, i, marker=MK[m], color=MC[m], ms=5.5, ls='none', zorder=3)
    ax.set_yticks(range(len(r))); ax.set_yticklabels([SHORT.get(m, m) for m in r.index]); ax.invert_yaxis()
    b0 = r.iloc[0]; ax.plot([b0, b0 + cd_all], [-1.1, -1.1], color='black', lw=2.2, solid_capstyle='butt', clip_on=False)
    ax.text(b0 + cd_all + 0.12, -1.1, f'DC = {cd_all:.2f}', va='center', fontsize=6.5)
    ax.set_ylim(len(r) - 0.4, -1.7); ax.set_xlim(3, 9)
    ax.set_xlabel('Rango promedio de la WQL por serie (1 = mejor)')
    T(ax, f'Rango promedio en {fmt(NS)} series'); panel_letter(ax, 'a', dx=-0.42)
    show = ['LightGBM global', 'Ensamble (mediana)', 'TimesFM-2.5', 'Chronos-2', 'N-HiTS global', 'ETS amortiguado']
    # b: horizonte
    ax = axs[0, 1]
    ax.axhline(0, color=MC['Log-lineal'], ls='--', lw=0.9)
    for m in show: ax.plot([1, 2, 3], (hor.loc[m].values - 1) * 100, marker=MK[m], ms=4, color=MC[m], lw=1.4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['1-12', '13-24', '25-36']); ax.set_xlim(0.8, 3.2)
    ax.set_xlabel('Horizonte (meses)'); ax.set_ylabel('WQL vs log-lineal (%)')
    ax.text(3.18, 0.5, 'log-lineal', color=MC['Log-lineal'], fontsize=6.5, ha='right', va='bottom')
    T(ax, 'La ventaja crece con el horizonte'); panel_letter(ax, 'b')
    # c: nivel
    ax = axs[1, 0]; lv = ['Nacional', 'Nacional×edad', 'Estatal', 'Estatal×edad']
    ax.axhline(0, color=MC['Log-lineal'], ls='--', lw=0.9)
    for m in show:
        if m in by_level.index: ax.plot(range(4), (by_level.loc[m, lv].values - 1) * 100, marker=MK[m], ms=4, color=MC[m], lw=1.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(['Nacional', 'Nacional\n× edad', 'Estatal', 'Estatal\n× edad']); ax.set_xlim(-0.25, 3.25)
    ax.set_ylabel('WQL vs log-lineal (%)'); ax.set_xlabel('Nivel de desagregación')
    T(ax, 'Mayor ganancia en series desagregadas'); panel_letter(ax, 'c', dx=-0.42)
    # d: exactitud vs calibración
    ax = axs[1, 1]; ax.axhline(80, color=ZERO, lw=0.7, ls=':'); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    lab = {'LightGBM global': (-1.5, 0, 'right', 'center'), 'Ensamble (mediana)': (-1.5, 0.3, 'right', 'center'),
           'TimesFM-2.5': (-1.5, -0.5, 'right', 'center'), 'Chronos-2': (-0.6, 1.3, 'right', 'bottom'),
           'Chronos-Bolt': (0.6, 1.3, 'left', 'bottom'), 'N-HiTS global': (1.5, -0.6, 'left', 'center'),
           'Log-lineal': (1.5, 0, 'left', 'center'), 'ETS amortiguado': (1.5, 0, 'left', 'center'),
           'SARIMA': (0, 1.4, 'center', 'bottom'), 'Ingenuo estacional': (0, -1.6, 'center', 'top')}
    for m in ci.index:
        x, y = (ci.loc[m, 'rel'] - 1) * 100, cov[m]
        ax.plot(x, y, marker=MK[m], color=MC[m], ms=5.5, ls='none', zorder=3)
        dx, dy, ha, va = lab[m]; ax.text(x + dx, y + dy, SHORT.get(m, m), color=MC[m], fontsize=6.5, ha=ha, va=va)
    ax.set_xlim(-24, 17); ax.set_ylim(58, 90)
    ax.set_xlabel('WQL vs log-lineal (%)'); ax.set_ylabel('Cobertura del IP 80 % (%)')
    T(ax, 'Exactitud frente a calibración'); panel_letter(ax, 'd', dx=-0.3)
    _family_legend(fig, show, y=0.0, ncol=6)
    _save(fig, out)

# ---------------------------------------------------------------- Figura 2
def fig2(plac_m, nat, out='fig_validacion_exceso.png'):
    fig = plt.figure(figsize=(W, 6.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.5, wspace=0.62)
    models = ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5',
              'LightGBM global', 'N-HiTS global', 'Ensamble (mediana)']
    ax = fig.add_subplot(gs[0, 0]); ax.axhline(10, color=ZERO, lw=0.7, ls=':')
    ax.text(59.5, 10.25, 'nominal 10 %', fontsize=6.5, ha='right', va='bottom', color=ZERO)
    lab = {'LightGBM global': (0, 0.45, 'center', 'bottom'), 'TimesFM-2.5': (0.9, 0.2, 'left', 'bottom'),
           'Chronos-Bolt': (0.9, 0.05, 'left', 'center'), 'Chronos-2': (0.9, -0.1, 'left', 'center'),
           'N-HiTS global': (-0.9, 0.0, 'right', 'center'), 'Ensamble (mediana)': (-0.9, -0.35, 'right', 'center'),
           'Ingenuo estacional': (0.3, -0.4, 'center', 'top'), 'ETS amortiguado': (0.9, 0, 'left', 'center'),
           'Log-lineal': (0.9, 0, 'left', 'center'), 'SARIMA': (0, 0.45, 'center', 'bottom')}
    for m in models:
        x, y = plac_m.loc[m, 'semiancho_%'], plac_m.loc[m, 'FPR_%']
        ax.plot(x, y, marker=MK[m], color=MC[m], ms=5, ls='none', zorder=3)
        dx, dy, ha, va = lab[m]; ax.text(x + dx, y + dy, SHORT.get(m, m), color=MC[m], fontsize=6, ha=ha, va=va)
    ax.set_xlim(22, 60); ax.set_ylim(3, 12)
    ax.set_xlabel('Semiancho del intervalo conformal (%)'); ax.set_ylabel('Falsos positivos (%)')
    T(ax, 'Placebo 2017-2019'); panel_letter(ax, 'a')
    # b: controles positivos
    ax = fig.add_subplot(gs[0, 1])
    pcs = [('Todas las causas', 1), ('Todas las causas', 2), ('Diabetes mellitus', 1), ('Diabetes mellitus', 2),
           ('Cardiopatía isquémica', 1), ('Cardiopatía isquémica', 2), ('Influenza y neumonía', 1), ('Influenza y neumonía', 2)]
    km = ['Log-lineal', 'Chronos-2', 'TimesFM-2.5', 'LightGBM global', 'Ensamble (mediana)']
    for i, (g, s) in enumerate(pcs):
        d = nat[(nat.grp == g) & (nat.sex == s) & (nat.year == 2020)].set_index('model'); e = d.loc['Ensamble (mediana)']
        ax.plot([e.lo, e.hi], [i, i], color='0.55', lw=3.5, solid_capstyle='butt', zorder=1, alpha=0.5)
        for j, m in enumerate(km): ax.plot(d.loc[m, 'pscore'], i, marker=MK[m], color=MC[m], ms=4, ls='none', zorder=3, alpha=0.95)
    ax.axvline(0, color=ZERO, lw=0.6)
    nm = {'Todas las causas': 'Todas las causas', 'Diabetes mellitus': 'Diabetes', 'Cardiopatía isquémica': 'Cardiop. isquémica',
          'Influenza y neumonía': 'Influenza y neumonía'}
    ax.set_yticks(range(len(pcs))); ax.set_yticklabels([f"{nm[g]} ({'H' if s == 1 else 'M'})" for g, s in pcs]); ax.invert_yaxis()
    ax.set_xlabel('Exceso en 2020 vs esperado (%)'); ax.set_xlim(-5, 135)
    T(ax, 'Controles positivos (2020)'); panel_letter(ax, 'b', dx=-0.62)
    h = [Line2D([], [], color=MC[m], marker=MK[m], ms=4, ls='none') for m in km] + [Line2D([], [], color='0.55', lw=3.5, alpha=0.5)]
    ax.legend(h, [SHORT.get(m, m) for m in km] + ['IC 90 % (ensamble)'], loc='upper right', frameon=False, fontsize=6, handletextpad=0.3,
              borderaxespad=0.2)
    # c: atlas de cánceres (mancuernas H/M)
    ax = fig.add_subplot(gs[1, :])
    a = nat[(nat.tipo == 'Cánceres') & (nat.model == 'Ensamble (mediana)') & (nat.year == 2022)].copy()
    a['site'] = a.grp.str.replace('C. ', '', regex=False).str.capitalize().str.replace('snc', 'SNC')
    order = a.groupby('site').pscore.mean().sort_values().index.tolist()
    SX = {1: ('Hombres', '#1F4E9C', 's', 0.17), 2: ('Mujeres', '#B2182B', 'o', -0.17)}
    for i, st in enumerate(order):
        for s, (nm_, c, mk, off) in SX.items():
            r = a[(a.site == st) & (a.sex == s)]
            if len(r):
                r = r.iloc[0]; ax.errorbar(i + off, r.pscore, yerr=[[r.pscore - r.lo], [r.hi - r.pscore]], fmt=mk, color=c, ms=4,
                                           capsize=1.8, lw=0.9, zorder=3)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha='right', rotation_mode='anchor')
    for tl in ax.get_xticklabels():
        if tl.get_text() == 'Mama': tl.set_fontweight('bold')
        if tl.get_text() in ('Leucemias', 'Linfomas y mieloma'): tl.set_fontweight('bold')
    ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylabel('Diferencia 2022 vs esperado (%)')
    nd = int((a.pscore < 0).sum())
    ax.legend([Line2D([], [], color=c, marker=mk, ls='none', ms=4) for _, (_, c, mk, _) in SX.items()], [v[0] for v in SX.values()],
              loc='upper right', frameon=False, ncol=2)
    T(ax, f'Cánceres en 2022: {nd} de {len(a)} series nacionales por debajo de lo esperado (ensamble, IC conformal 90 %)')
    panel_letter(ax, 'c', dx=-0.07)
    _save(fig, out)

# ---------------------------------------------------------------- Figura 3
def fig3(vals, seeds, cmp, wv, nP, bvals, nb, out='fig_adaptacion_LoRA.png'):
    cols = ['#666666', '#D6604D', '#67001F', '#B2182B']
    labs = ['Ajuste completo, 1 000 pasos', 'LoRA, 300 pasos', 'LoRA, 1 000 pasos (3 semillas)', 'LoRA, 1 000 pasos, sin mama']
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.3), gridspec_kw=dict(width_ratios=[1.25, 0.85, 0.9], wspace=0.35))
    ax = axs[0]; y = np.arange(4)
    ax.barh(y, (np.array(vals) - 1) * 100, color=cols, height=0.6)
    ax.plot((np.array(seeds) - 1) * 100, [2] * 3, 'o', mfc='white', mec='black', ms=3.5, mew=0.7, zorder=3)
    for m, ls in [('LightGBM global', '--'), ('Ensamble (mediana)', ':')]:
        ax.axvline((cmp[m] - 1) * 100, color=MC[m], ls=ls, lw=1.1)
    ax.text((cmp['LightGBM global'] - 1) * 100 + 0.05, 3.75, 'LightGBM', fontsize=6.5, color=MC['LightGBM global'], ha='left', va='center')
    ax.text((cmp['Ensamble (mediana)'] - 1) * 100 - 0.05, 3.75, 'Ensamble', fontsize=6.5, color='black', ha='right', va='center')
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_ylim(4.1, -0.6); ax.set_xlim(-3.4, 0)
    ax.set_xlabel('WQL vs Chronos-2 sin adaptar (%)'); T(ax, 'Error relativo'); panel_letter(ax, 'a', dx=-1.0)
    ax = axs[1]; ax.barh(y, wv, color=cols, height=0.6); ax.axvline(50, color=ZERO, lw=0.7, ls=':')
    for i, v in enumerate(wv): ax.text(v + 0.8, i, f'{v:.0f} %', va='center', fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_ylim(4.1, -0.6); ax.set_xlim(40, 78)
    ax.set_xlabel(f'% de {fmt(nP)} series (p < 0.001)'); T(ax, 'Series que mejoran'); panel_letter(ax, 'b', dx=-0.12)
    ax = axs[2]; bc = ['#D55E00', '#67001F', '#B2182B']
    ax.bar(range(3), (np.array(bvals) - 1) * 100, color=bc, width=0.6); ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Sin\nadaptar', 'LoRA\n(todas)', 'LoRA\n(sin mama)'])
    ax.set_ylabel('WQL vs sin adaptar (%)'); ax.set_ylim(-4.2, 0.4)
    for i, v in enumerate((np.array(bvals) - 1) * 100):
        if i: ax.text(i, v - 0.12, f'{v:.1f} %'.replace('-', '−'), ha='center', va='top', fontsize=6.5)
    T(ax, f'Cáncer de mama (n = {nb} series)'); panel_letter(ax, 'c', dx=-0.3)
    _save(fig, out)

# ---------------------------------------------------------------- Figura 4
def fig4(out='fig1_contrafactual_TEE_C50.png'):
    fc = pd.read_csv('tsfm_contrafactual_mensual.csv', parse_dates=['fecha']); fc['model'] = fc.model.replace(REN)
    Sx = pd.read_csv('series_mensuales_C50.csv', parse_dates=['fecha']).set_index('fecha')
    bm = pd.read_csv('tsfm_backtest_mensual.csv'); bm['model'] = bm.model.replace(REN)
    ex2 = pd.read_csv('tsfm_exceso_anual.csv'); ex2['model'] = ex2.model.replace(REN)
    order = ['Ingenuo estacional', 'Log-lineal', 'SARIMA', 'ETS amortiguado', 'Chronos-2', 'Chronos-Bolt', 'TimesFM-2.5', 'Ensamble (mediana)']
    fig = plt.figure(figsize=(W, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.1], hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(gs[0, :]); g = fc[(fc.target == 'asmr') & (fc.model == 'Ensamble (mediana)')]
    ax.fill_between(g.fecha, g.q10, g.q90, color='#F4B183', alpha=0.55, lw=0, label='Intervalo de 80 % (ensamble)')
    ax.plot(Sx.index, Sx.asmr, color='black', lw=0.9, label='Observada')
    ax.plot(g.fecha, g.pred, color='#D55E00', lw=1.6, label='Esperada sin pandemia (ensamble)')
    ax.axvline(pd.Timestamp('2020-01-01'), color=ZERO, lw=0.7, ls=':')
    ax.set_ylabel('TEE por 100 000\nmujeres-año'); ax.set_ylim(13.5, 22.5); ax.margins(x=0.01)
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
    ax = fig.add_subplot(gs[1, 1]); e = ex2[ex2.target == 'asmr']
    ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for j, m in enumerate(order[:-1]):
        gg = e[e.model == m]; ax.plot(gg.year + (j - 3) * 0.075 - 0.08, gg.pscore, marker=MK[m], color=MC[m], ms=2.8, ls='none', zorder=3)
    gg = e[e.model == 'Ensamble (mediana)']
    ax.errorbar(gg.year + 0.28, gg.pscore, yerr=[gg.pscore - gg.p_lo, gg.p_hi - gg.pscore], fmt='D', color='black', ms=3.5, capsize=2,
                lw=0.9, zorder=4)
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_xticks(range(2020, 2025))
    ax.set_xticklabels(["'20", "'21", "'22", "'23", "'24"]); ax.set_ylim(-17, 12)
    T(ax, 'Por modelo'); panel_letter(ax, 'c', dx=-0.3)
    ax = fig.add_subplot(gs[1, 2]); AC = {'asmr_20-39': ('20-39', '#3B8BC2', 'o'), 'asmr_40-59': ('40-59', '#1F4E9C', 's'), 'asmr_60+': ('≥ 60', '#0B1F3F', 'D')}
    ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (tg, (lb, cc, mk)) in enumerate(AC.items()):
        gg = ex2[(ex2.target == tg) & (ex2.model == 'Ensamble (mediana)')]
        ax.errorbar(gg.year + (k - 1) * 0.22, gg.pscore, yerr=[gg.pscore - gg.p_lo, gg.p_hi - gg.pscore], fmt=mk, color=cc, ms=3,
                    capsize=1.5, lw=0.8, label=lb + ' años')
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(["'20", "'21", "'22", "'23", "'24"])
    ax.set_ylim(-24, 30); ax.set_ylabel('Diferencia vs esperado (%)')
    ax.legend(frameon=False, loc='upper right', fontsize=6, handletextpad=0.2, borderaxespad=0.1)
    T(ax, 'Por edad (ensamble)'); panel_letter(ax, 'd', dx=-0.3)
    _family_legend(fig, order, y=-0.035, ncol=8)
    _save(fig, out)

# ---------------------------------------------------------------- Figura S1
def figS1(A, def_pt, out='fig2_subgrupos_cosecha_C50.png'):
    es = pd.read_csv('tsfm_exceso_anual_subgrupos.csv'); E = es[es.model == 'Ensamble (mediana)']
    pl = pd.read_csv('lugar_defuncion_C50.csv', index_col=0)
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.6), gridspec_kw=dict(hspace=0.5, wspace=0.35))
    RC = {'Norte': ('#1B9E77', 'o'), 'Centro-Occidente': ('#D95F02', 's'), 'Centro': ('#7570B3', '^'), 'Sur-Sureste': ('#C2185B', 'D')}
    ax = axs[0, 0]; ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (r, (cc, mk)) in enumerate(RC.items()):
        g = E[E.target == f'asmr_{r}']
        ax.errorbar(g.year + (k - 1.5) * 0.17, g.pscore, yerr=[g.pscore - g.p_lo, g.p_hi - g.pscore], fmt=mk, ms=3, color=cc, capsize=1.5, lw=0.8, label=r)
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-38, 24)
    ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=6, handletextpad=0.2, columnspacing=0.8)
    T(ax, 'Por región (TEE)'); panel_letter(ax, 'a')
    ax = axs[0, 1]; ax.axvspan(2023.5, 2024.5, color=SHADE, lw=0)
    for k, (tg, (lab, cc, mk)) in enumerate({'deaths_SS': ('Con seguridad social', '#1F4E9C', 's'), 'deaths_noSS': ('Sin seguridad social', '#A6611A', 'o')}.items()):
        g = E[E.target == tg]
        ax.errorbar(g.year + (k - 0.5) * 0.24, g.pscore, yerr=[g.pscore - g.p_lo, g.p_hi - g.pscore], fmt=mk, ms=3.2, color=cc, capsize=1.5, lw=0.8, label=lab)
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-38, 24)
    ax.legend(frameon=False, loc='lower left', fontsize=6)
    T(ax, 'Por derechohabiencia (conteos)'); panel_letter(ax, 'b')
    ax = axs[1, 0]; ax.hist(A, bins=60, color='#5E35B1', lw=0); ax.axvline(def_pt, color='black', lw=1.2)
    ax.text(def_pt - 40, ax.get_ylim()[1] * 0.95, f'Déficit observado\n2021-2024: {fmt(int(round(def_pt)))}', fontsize=6.5, va='top', ha='right')
    ax.set_xlabel('Defunciones por cáncer de mama adelantadas por COVID-19'); ax.set_ylabel('Simulaciones'); ax.set_yticks([])
    ax.set_xlim(0, 2600); T(ax, 'Simulación de cosecha'); panel_letter(ax, 'c')
    ax = axs[1, 1]; pl2 = pl.loc[2012:2024]; ax.axvspan(2019.5, 2021.5, color=SHADE, lw=0)
    ax.plot(pl2.index, pl2['Hogar'], color='#B2182B', marker='o', ms=3, lw=1.3)
    ax.plot(pl2.index, pl2['Unidad pública'], color='#1F4E9C', marker='s', ms=3, lw=1.3)
    ax.text(2012, pl2['Hogar'].iloc[0] + 3, 'Hogar', color='#B2182B', fontsize=7)
    ax.text(2012, pl2['Unidad pública'].iloc[0] + 3, 'Unidad médica pública', color='#1F4E9C', fontsize=7)
    ax.set_ylabel('Defunciones (%)'); ax.set_ylim(15, 72); ax.set_xticks(range(2012, 2025, 2))
    T(ax, 'Lugar de defunción'); panel_letter(ax, 'd')
    _save(fig, out)

# ---------------------------------------------------------------- Figura S2
def figS2(out='fig_subregistro.png'):
    lab = {'age': 'Edad', 'esc_g': 'Escolaridad', 'sitio_g': 'Lugar de defunción', 'ent_g': 'Entidad', 'tloc_g': 'Tamaño de localidad',
           'asist_g': 'Asistencia médica', 'civ_g': 'Estado civil', 'dh_g': 'Derechohabiencia', 'm': 'Mes'}
    cal = pd.read_csv('subregistro_calibracion.csv'); perf = pd.read_csv('subregistro_desempeno.csv').iloc[0]
    sh = pd.read_csv('subregistro_shap.csv', index_col=0).iloc[:, 0].sort_values()
    tot = pd.read_csv('subregistro_resultados.csv', index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 0.9], wspace=0.95))
    ax = axs[0]; mx = float(max(cal.pred.max(), cal.obs.max())) * 1.1
    ax.plot([0, mx], [0, mx], color='0.55', lw=0.8, ls='--'); ax.plot(cal.pred, cal.obs, 'o-', color='#5E35B1', ms=3.5, lw=1.2)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Proporción observada')
    ax.text(0.04, 0.96, f"AUC {perf.AUC:.2f}\n(solo edad {perf.AUC_edad:.2f})", transform=ax.transAxes, va='top', fontsize=6.5)
    T(ax, 'Calibración, 2019'); panel_letter(ax, 'a', dx=-0.35)
    ax = axs[1]; ax.barh(range(len(sh)), sh.values, color='#5E35B1', height=0.6)
    ax.set_yticks(range(len(sh))); ax.set_yticklabels([lab.get(i, i) for i in sh.index]); ax.set_xlabel('|SHAP| medio (log-odds)')
    T(ax, 'Importancia de variables'); panel_letter(ax, 'b', dx=-0.8)
    ax = axs[2]; ex_ = tot.loc[2020:2024, 'exceso_vs_2017_19']
    ax.bar(ex_.index, ex_.values, color='#5E35B1', width=0.6)
    ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(["'20", "'21", "'22", "'23", "'24"]); ax.set_ylim(0, 34)
    ax.set_ylabel('Defunciones por año')
    ax.text(0.55, 0.97, f'2021-2024: {ex_.loc[2021:2024].sum():.0f} de 1 954\n(~{ex_.loc[2021:2024].sum() / 1954 * 100:.0f} % del déficit)',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    T(ax, 'C50 en causas mal definidas'); panel_letter(ax, 'c', dx=-0.35)
    _save(fig, out)

# ---------------------------------------------------------------- Figura S3
def figS3(out='fig_finetune.png'):
    FC = {'Chronos-2 zero-shot': '#D55E00', 'Chronos-2 + contexto entre series': '#A6611A', 'Chronos-2 LoRA (México)': '#67001F',
          'Chronos-2 LoRA + contexto': '#C2185B', 'Log-lineal': '#1F4E9C'}
    relv = pd.read_csv('finetune_rel_por_volumen.csv', index_col=0); FA = pd.read_csv('finetune_resultados.csv.gz')
    cf = pd.read_csv('finetune_contrafactual_C50.csv'); ens = pd.read_csv('tsfm_exceso_anual.csv')
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
    ax.barh(range(3), wr, color=[FC[m] for m in mm], height=0.6); ax.axvline(50, color=ZERO, lw=0.7, ls=':')
    ax.set_yticks(range(3)); ax.set_yticklabels(['Contexto', 'LoRA', 'LoRA +\ncontexto']); ax.invert_yaxis()
    for i, v in enumerate(wr): ax.text(v + 1, i, f'{v:.0f} %', va='center', fontsize=6.5)
    ax.set_xlim(40, 80); ax.set_xlabel('Series que mejoran (%)'); T(ax, f'n = {fmt(len(Pw))}; p < 0.001'); panel_letter(ax, 'b', dx=-0.45)
    ax = axs[2]; c1 = cf[cf.target == 'asmr'].pivot(index='year', columns='model', values='pscore')
    e0 = ens[(ens.target == 'asmr') & (ens.model == 'Ensamble (mediana)')].set_index('year')
    ax.fill_between(e0.index.astype(int), e0.p_lo, e0.p_hi, color='0.85', lw=0, label='IC 90 % (ensamble)')
    ax.plot(c1.index, c1['Chronos-2 zero-shot'], 'o-', color=FC['Chronos-2 zero-shot'], ms=3, lw=1.2, label='Zero-shot')
    ax.plot(c1.index, c1['Chronos-2 LoRA (México)'], 's-', color=FC['Chronos-2 LoRA (México)'], ms=3, lw=1.2, label='LoRA')
    ax.axhline(0, color=ZERO, lw=0.6); ax.set_xticks(range(2020, 2025)); ax.set_xticklabels(["'20", "'21", "'22", "'23", "'24"])
    ax.set_ylabel('Diferencia vs esperado (%)'); ax.set_ylim(-17, 11)
    ax.legend(frameon=False, fontsize=6, loc='lower left'); T(ax, 'Cáncer de mama (TEE)'); panel_letter(ax, 'c', dx=-0.35)
    _save(fig, out)
