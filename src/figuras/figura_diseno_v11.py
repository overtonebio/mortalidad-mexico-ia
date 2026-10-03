"""Figura 1: periodos originales y ubicación explícita de los análisis exploratorios.
Diagrama, sin cambios a ventanas, datos ni resultados del pronóstico.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figuras_principales'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':8,'xtick.labelsize':7,'axes.linewidth':.6,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'ps.fonttype':42,'savefig.dpi':300})
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=.43)
ax = fig.add_subplot(gs[0]); ax.set_xlim(2012, 2025.05); ax.set_ylim(-.6, 3.9)
rows=[('Inicio dic-2014',2015,2018,2019),('Inicio dic-2015',2016,2019,2020),
      ('Inicio dic-2016',2017,2020,None),('Inicio dic-2019',2020,2024,2025)]
for i,(lab,a,b,c) in enumerate(rows):
    y=3-i; col='#F4B183' if i<3 else '#D55E00'
    ax.add_patch(Rectangle((2012,y-.30),a-2012,.6,color='#BFD3E6',lw=0))
    ax.add_patch(Rectangle((a,y-.30),b-a,.6,color=col,lw=0))
    if c: ax.add_patch(Rectangle((b,y-.30),c-b,.6,facecolor='white',edgecolor=col,hatch='////',lw=.8))
    ax.text(2011.85,y,lab,ha='right',va='center',fontsize=7)
ax.text(2013.5,3,'Entrenamiento',ha='center',va='center',fontsize=6.5)
ax.text(2016.5,3,'Prueba, 36 meses',ha='center',va='center',fontsize=6.5)
ax.text(2018.5,3.45,'4.º año (48 meses)',ha='center',va='bottom',fontsize=6.5)
ax.text(2016,0,'Entrenamiento: 2012-2019',ha='center',va='center',fontsize=6.5)
ax.text(2022,0,'Esperado 2020-2023',ha='center',va='center',fontsize=6.5,color='white')
ax.text(2024.5,.45,'2024: sin\nvalidar',ha='center',va='bottom',fontsize=6.5)
ax.axvline(2020,color='.45',lw=.7,ls=':');ax.set_yticks([]);ax.spines['left'].set_visible(False)
ax.set_xticks(range(2012,2026,2));ax.set_title('Periodos de entrenamiento y prueba',pad=10)
ax.text(-.19,1.03,'a',transform=ax.transAxes,fontweight='bold',fontsize=10)
ax=fig.add_subplot(gs[1]);ax.set_xlim(0,100);ax.set_ylim(0,64);ax.axis('off')
boxes=[(1,'Datos','Defunciones\nINEGI 2012-2024\n2 870 series\npor causa, sexo,\nentidad y edad','#E8EEF7'),
       (21,'Pronóstico','10 métodos:\n4 clásicos,\n5 de IA\ny combinación\nde seis modelos','#E8EEF7'),
       (41,'Exactitud','Error del\npronóstico\n(WQL)\ny cobertura\nde intervalos','#FBE5D6'),
       (61,'Detección','Falsas alarmas\nen 2017-2019\ny aumentos\nsimulados de\n10 % y 20 %','#FBE5D6'),
       (81,'Cáncer de mama','Observado frente\na esperado,\ncon intervalos\nde predicción\nde 90 %','#F8CBAD')]
for x,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,24),17.5,34,boxstyle='round,pad=.35,rounding_size=1',fc=c,ec='.4',lw=.7))
    ax.text(x+8.75,54.5,h,ha='center',va='top',fontsize=7.2,fontweight='bold')
    ax.text(x+8.75,47.8,t,ha='center',va='top',fontsize=6.8,linespacing=1.4)
for x in (19.2,39.2,59.2,79.2):
    ax.annotate('',xy=(x+1.6,41),xytext=(x-.2,41),arrowprops=dict(arrowstyle='-|>',color='.3',lw=.9))
ax.add_patch(FancyBboxPatch((42,1),56,15.5,boxstyle='round,pad=.35,rounding_size=1',fc='#EEF4F2',ec='.4',lw=.7))
ax.text(70,13.5,'Análisis exploratorios de cáncer de mama',ha='center',va='top',fontsize=7.2,fontweight='bold')
ax.text(70,8.8,'Tendencia desde marzo de 2020 y lugar de defunción\nDatos 2012-2023; 2024 como sensibilidad',ha='center',va='top',fontsize=6.8,linespacing=1.35)
ax.annotate('',xy=(89.5,17.1),xytext=(89.5,23.2),arrowprops=dict(arrowstyle='-|>',color='.3',lw=.9))
ax.text(39,8.5,'Posteriores al\nanálisis principal',ha='right',va='center',fontsize=6.8,color='.3')
ax.set_title('Flujo del estudio',pad=6)
ax.text(-.02,1.01,'b',transform=ax.transAxes,fontweight='bold',fontsize=10)
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'Figura_1_v11.{ext}',bbox_inches='tight',pad_inches=.06)
plt.close(fig)
