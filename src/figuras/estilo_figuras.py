"""Parámetros gráficos comunes (independientes del entorno): escala tipográfica 8/7/6 pt, ejes abiertos, 300 ppp."""
import matplotlib as mpl
def apply_figure_style(sizes=(8, 7, 6)):
    base, sec, tick = sizes
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': base, 'axes.labelsize': base,
                         'axes.titlesize': base, 'legend.fontsize': sec, 'xtick.labelsize': tick, 'ytick.labelsize': tick, 'axes.linewidth': 0.6,
                         'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': False, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'lines.linewidth': 1.2, 'patch.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter.lower(), transform=ax.transAxes, fontweight='bold', fontsize=mpl.rcParams['font.size'] + 1, va='bottom', ha='left')
