"""Rebuild Utility tables and figures from the portable saved-report extracts."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1]
FIG = OUT / 'figures'
TAB = OUT / 'tables'
COLORS = {'B0':'#D98932','B1':'#277DA8','B2':'#8C4F9F','Real':'#222222'}
FAMILIES = ['LOGBANDPOWER_LOGREG', 'OAS_RIEMANN_LINEAR_SVM', 'EEGNETV4']
TITLES = ['LogBandpower + LR', 'Riemannian + SVM', 'EEGNetv4']
CONDITIONS = ['REAL_ONLY', 'REAL_DUPLICATE_1X_CONTROL', 'TSTR_B0_1X',
              'TSTR_B1_1X', 'TSTR_B2_1X', 'REAL_PLUS_B0_0P5X',
              'REAL_PLUS_B0_1X', 'REAL_PLUS_B1_0P5X', 'REAL_PLUS_B1_1X',
              'REAL_PLUS_B2_0P5X', 'REAL_PLUS_B2_1X']
LABELS = ['Real only','Real duplicate','TSTR B0','TSTR B1','TSTR B2',
          'Real + B0 0.5x','Real + B0 1x','Real + B1 0.5x','Real + B1 1x',
          'Real + B2 0.5x','Real + B2 1x']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})


def read(name):
    with (TAB / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for suffix in ('pdf','png'):
        fig.savefig(FIG / f'{name}.{suffix}', bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def color(cond):
    return next((COLORS[b] for b in ['B0','B1','B2'] if b in cond), COLORS['Real'])


summary = read('utility_summary.csv')
contrasts = read('utility_contrasts.csv')
learning = read('utility_learning_curves.csv')
lookup = {(r['family'], r['condition']):r for r in summary}

# Actual LaTeX table, no rasterized text. Numbers stay left-to-right in xepersian.
latex = [r'\begin{tabular}{lccc}', r'\toprule',
         r'\lr{Condition} & \lr{LogBP + LR} & \lr{Riemann + SVM} & \lr{EEGNetv4} \\',
         r'\midrule']
for i, (c, label) in enumerate(zip(CONDITIONS, LABELS)):
    vals = []
    for f in FAMILIES:
        r = lookup[f,c]
        v = f"{float(r['test_mean_percent']):.3f}"
        if f == 'EEGNETV4':
            v += r' $\pm$ ' + f"{float(r['test_population_sd_pp']):.3f}"
        vals.append(r'\lr{' + v + '}')
    latex.append(r'\lr{' + label.replace('0.5x', r'0.5$\times$').replace(' 1x', r' 1$\times$') + '} & ' + ' & '.join(vals) + r' \\')
    if i in (1,4):
        latex.append(r'\addlinespace[2pt]')
latex += [r'\bottomrule', r'\end{tabular}']
(TAB / 'final_test_table.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')

# Three compact panels retain all 11 conditions and all classifier families.
fig, axes = plt.subplots(1,3,figsize=(7.35,4.1),sharey=True)
for ax, family, title in zip(axes,FAMILIES,TITLES):
    for y,c in enumerate(CONDITIONS):
        r = lookup[family,c]
        co = color(c)
        for split,dy,fill in [('validation',-.13,'white'),('test',.13,co)]:
            m = float(r[f'{split}_mean_percent'])
            sd = float(r[f'{split}_population_sd_pp'])
            ax.errorbar(m,y+dy,xerr=sd if family=='EEGNETV4' else None,
                        fmt='o',ms=4,mfc=fill,mec=co,color=co,elinewidth=1,capsize=2,zorder=3)
    ax.set_title(title,pad=9)
    ax.set_xlabel('Subject-macro BA (%)')
    ax.grid(axis='x',alpha=.18,zorder=0)
    ax.axhline(1.5,color='#bbbbbb',lw=.6)
    ax.axhline(4.5,color='#bbbbbb',lw=.6)
    ax.set_ylim(10.6,-.6)
    ax.set_yticks(range(11),LABELS)
axes[0].set_xlim(79.5,85.1)
axes[1].set_xlim(75.9,82.4)
axes[2].set_xlim(83.1,90.4)
fig.legend([Line2D([],[],marker='o',ls='',mfc='white',mec='#333'),
            Line2D([],[],marker='o',ls='',color='#333')],
           ['Validation (development)','Final Test'],loc='lower center',bbox_to_anchor=(.58,-.025),ncol=2,frameon=False,fontsize=9)
fig.subplots_adjust(left=.19,right=.99,bottom=.17,top=.91,wspace=.25)
save(fig,'utility_validation_test')

# Stored paired bootstrap intervals only; no new contrast, p-value or resampling.
entries = [('EEGNETV4','S6_AUG_B0_0P5X_VS_REAL','EEGNet: B0 0.5x - Real','B0'),
           ('EEGNETV4','S3_AUG_B0_1X_VS_REAL','EEGNet: B0 1x - Real','B0'),
           ('EEGNETV4','P2_AUG_B2_1X_VS_REAL','EEGNet: B2 1x - Real','B2'),
           ('EEGNETV4','P3_AUG_B2_1X_VS_B1_1X','EEGNet: B2 1x - B1 1x','B2'),
           ('EEGNETV4','P1_TSTR_B2_VS_B1','EEGNet: TSTR B2 - B1','B2'),
           ('LOGBANDPOWER_LOGREG','P2_AUG_B2_1X_VS_REAL','LogBP + LR: B2 1x - Real','B2'),
           ('OAS_RIEMANN_LINEAR_SVM','P2_AUG_B2_1X_VS_REAL','Riemann + SVM: B2 1x - Real','B2')]
fig, ax = plt.subplots(figsize=(7.35,3.2))
for y,(family,ident,label,key) in enumerate(entries):
    r=next(r for r in contrasts if r['family']==family and r['contrast_id']==ident)
    v,lo,hi=[float(r[k]) for k in ['delta_left_minus_right_pp','ci95_low_pp','ci95_high_pp']]
    ax.errorbar(v,y,xerr=np.array([[v-lo],[hi-v]]),fmt='o',ms=5,color=COLORS[key],capsize=3,lw=1.5)
    ax.text(2.02,y,f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]',va='center',fontsize=8.2)
ax.text(2.02,-.8,'Difference [95% CI]',va='center',fontsize=8.2)
ax.set_yticks(range(len(entries)),[x[2] for x in entries])
ax.set_ylim(6.6,-1)
ax.set_xlim(-1.65,1.9)
ax.axvline(0,color='#777777',lw=.8,ls='--')
ax.axhline(4.5,color='#bbbbbb',lw=.7)
ax.grid(axis='x',alpha=.15)
ax.set_xlabel('Difference in subject-macro BA (percentage points)')
fig.subplots_adjust(left=.29,right=.71,top=.97,bottom=.17)
save(fig,'utility_contrasts')

# Two illustrative conditions, with every seed and every observed epoch retained.
# Source is encoded by color, seed by line style. Circles mark the chosen epoch.
fig, axes=plt.subplots(2,2,figsize=(7.35,3.85),sharex=True)
styles=['-','--',':']
for col,cond in enumerate(['REAL_ONLY','REAL_PLUS_B2_1X']):
    for i,seed in enumerate([20262301,20262302,20262303]):
        rr=[r for r in learning if r['condition']==cond and int(r['seed'])==seed]
        ep=np.array([int(r['epoch']) for r in rr])
        best=int(rr[0]['best_epoch'])
        val=np.array([float(r['validation_ba_percent']) for r in rr])
        trainloss=np.array([float(r['train_loss']) for r in rr])
        valloss=np.array([float(r['validation_loss']) for r in rr])
        axes[0,col].plot(ep,val,styles[i],color=color(cond),lw=1.3)
        axes[0,col].plot(best,val[best-1],'o',color=color(cond),ms=4,mfc='white',mew=1)
        axes[1,col].plot(ep,valloss,styles[i],color=color(cond),lw=1.3)
        axes[1,col].plot(ep,trainloss,styles[i],color=color(cond),lw=.9,alpha=.3)
    axes[0,col].set_title(['Real only','Real + B2 1x'][col],pad=5)
    axes[1,col].set_xlabel('Epoch')
    for row in range(2):
        axes[row,col].grid(alpha=.17)
    axes[0,col].set_ylim(78,92)
    axes[1,col].set_ylim(.1,.65)
axes[0,0].set_ylabel('Validation BA (%)')
axes[1,0].set_ylabel('Cross-entropy')
fig.legend([Line2D([],[],color='#555',ls=s,lw=1.3) for s in styles] +
           [Line2D([],[],color='#555',ls='',marker='o',mfc='white')],
           ['Seed 1','Seed 2','Seed 3','Selected epoch'],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.53,-.025),fontsize=8)
fig.subplots_adjust(left=.1,right=.99,top=.92,bottom=.15,hspace=.16,wspace=.2)
save(fig,'utility_learning')
print('Wrote final_test_table.tex and Utility figures (PDF + PNG).')
