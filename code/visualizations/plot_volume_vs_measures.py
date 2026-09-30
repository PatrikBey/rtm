#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                GRAPH REPRESENTATION OF INTELLIGENCE                   #
#                                                                       #
# Each RT-derived behaviour measure against lesion volume, one subplot  #
# per measure, with the Foreperiod and Go/No-Go versions of that        #
# measure overlaid in the same panel and colour-coded.                  #
#                                                                       #
# Only measures present for BOTH tasks are plotted, so every panel is a #
# like-for-like comparison; the subplot title is the shared column name #
# with the task prefix stripped. The original per-subject ex-Gaussian   #
# tau is included as the first panel, the baseline everything else was  #
# built to replace. SATO is excluded: it has no remodelled counterpart. #
#                                                                       #
# BOTH axes are LOG-scaled and labelled as such. Lesion volume spans    #
# 485-376677 mm^3, three orders of magnitude, and on a linear axis the  #
# smaller lesions collapse onto the origin. The behaviour axis is the   #
# measure's own [0.001, 1] normalised scale, shared across panels; it   #
# is logged because several measures are bimodal, with a dense cluster  #
# pinned near the floor and a sparse upper cluster, and on a linear     #
# axis that split is invisible. Two log axes on one panel is not a      #
# dual-axis chart -- each variable still has exactly one scale.         #
#                                                                       #
# Colour: the same two fixed categorical slots used for these two tasks #
# elsewhere in the project (blue = Foreperiod, orange = Go/No-Go), so   #
# colour follows the entity across figures. Validated with the dataviz  #
# palette validator: all checks pass, worst adjacent CVD dE 24.7.       #
#                                                                       #
# usage: plot_volume_vs_measures.py --data_path /data/patrik/RT/RTM     #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


#################################
#      LOGGING UTILITIES        #
#################################

def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')


#################################
#       PARSE PARAMETERS        #
#################################

parser = argparse.ArgumentParser(
    description='Lesion volume vs. each paired Foreperiod/Go-No-Go behaviour measure.')
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv (default: {data_path}/participants.tsv)')
parser.add_argument('--volume_col', type=str, default='lesion_volume_mm3',
                    help='Column holding raw lesion volume (default: lesion_volume_mm3)')
parser.add_argument('--out_dir', type=str, default=None,
                    help='Output directory (default: {data_path}/FIGURES)')
parser.add_argument('--n_cols', type=int, default=3, help='Subplot grid width (default: 3)')
parser.add_argument('--no_original_tau', action='store_true', default=False,
                    help='Omit the original per-subject ex-Gaussian tau baseline panel')
parser.add_argument('--measures', type=str, nargs='+', default=None,
                    help='Explicit list of common measure names to plot (i.e. the '
                       'foreperiod_/gonogo_ suffix, e.g. hexg_total median_rt), instead of '
                       'auto-discovering every measure present for both tasks')
parser.add_argument('--x_measure', type=str, default=None,
                    help='Common measure name (foreperiod_/gonogo_ suffix, e.g. median_rt) to use '
                       'as the x-axis instead of --volume_col -- paired per task like the y-axis '
                       'measures, so each task\'s own points use that task\'s own column')
args = parser.parse_args()

participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')
out_dir = args.out_dir or os.path.join(args.data_path, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

# fixed categorical slots, same entity -> same colour as elsewhere in the project
COLOR = {'foreperiod': '#2a78d6', 'gonogo': '#eb6834'}
LABEL = {'foreperiod': 'Foreperiod', 'gonogo': 'Go/No-Go'}

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'text.color': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e5e4df', 'grid.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

log_msg(f"| START | Lesion volume vs. behaviour measures | {participants_path}")


#################################
#      LOAD + PAIR COLUMNS      #
#################################

df = pd.read_csv(participants_path, sep='\t')
if args.x_measure is None:
    volume = pd.to_numeric(df[args.volume_col], errors='coerce')
    x_for = {'foreperiod': volume, 'gonogo': volume}
    x_label = 'Lesion volume (mm³, log scale)'
else:
    x_for = {t: pd.to_numeric(df[f'{t}_{args.x_measure}'], errors='coerce') for t in ('foreperiod', 'gonogo')}
    x_label = f'{args.x_measure} (x/max, log scale)'

# the original ex-Gaussian fits predate the foreperiod_/gonogo_ naming
# convention, so they are paired explicitly rather than by prefix. They are the
# baseline every other measure was built to replace, hence plotted first.
ORIGINAL_TAU = ('original_exgaussian_tau', {'foreperiod': 'Foreperiod_Long_tau',
                                            'gonogo': 'GoNoGo_tau'})

# a measure is plotted only if BOTH tasks have it, so each panel compares like with like
fp = {c[len('foreperiod_'):] for c in df.columns if c.startswith('foreperiod_')}
gg = {c[len('gonogo_'):] for c in df.columns if c.startswith('gonogo_')}
measures = args.measures if args.measures is not None else sorted(fp & gg)

# name -> {task: column}, so panels can mix both naming conventions
columns_for = {m: {t: f'{t}_{m}' for t in ('foreperiod', 'gonogo')} for m in measures}
if not args.no_original_tau and all(c in df.columns for c in ORIGINAL_TAU[1].values()):
    columns_for = {ORIGINAL_TAU[0]: ORIGINAL_TAU[1], **columns_for}
    measures = [ORIGINAL_TAU[0]] + measures
    log_msg(f"| UPDATE | Baseline panel added: {ORIGINAL_TAU[1]}")
log_msg(f"| UPDATE | {len(measures)} measures present for both tasks: {measures}")
dropped = (fp ^ gg)
if dropped:
    log_msg(f"| WARNING | Skipped {len(dropped)} measure(s) present for only one task: {sorted(dropped)}")


#################################
#        SCATTER GRID           #
#################################

n_cols = args.n_cols
n_rows = int(np.ceil(len(measures) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows))
axes = np.atleast_1d(axes).ravel()

for ax, measure in zip(axes, measures):
    for task in ('foreperiod', 'gonogo'):
        x = x_for[task]
        y = pd.to_numeric(df[columns_for[measure][task]], errors='coerce')
        ok = y.notna() & x.notna()
        ax.scatter(x[ok], y[ok], s=20, color=COLOR[task], alpha=0.55,
                   edgecolors='none', label=LABEL[task])
        rho, p = spearmanr(x[ok], y[ok])
        # rho is reported per panel because the question these panels answer is
        # whether the x-axis measure tracks the y-axis measure at all, which the
        # cloud alone does not settle at this density
        # a background box is required, not decorative: these labels sit on top of
        # dense point clouds in several panels and are unreadable without it
        ax.text(0.03, 0.97 - 0.10 * (task == 'gonogo'), f'ρ={rho:+.2f}' + ('*' if p < 0.05 else ''),
                transform=ax.transAxes, fontsize=8.5, color=COLOR[task],
                va='top', ha='left', fontweight='bold', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.82,
                          boxstyle='round,pad=0.22'))
    ax.set_xscale('log')
    # the behaviour axis is log-scaled too: several measures -- the original
    # ex-Gaussian tau above all -- are bimodal, with a dense cluster pinned near
    # the bottom of the [0.001, 1] range and a sparse upper cluster. On a linear
    # axis one extreme value compresses the lower cluster into a flat line and
    # the split is invisible. The 0.001 floor (rather than 0) is what makes this
    # possible: log of an exact zero is undefined.
    ax.set_yscale('log')
    ax.set_title(measure, fontsize=10.5, fontweight='bold')
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.15)

for ax in axes[len(measures):]:
    ax.set_visible(False)

# one shared axis label per edge rather than repeating it on every panel
for i, ax in enumerate(axes[:len(measures)]):
    if i % n_cols == 0:
        ax.set_ylabel('Measure value (x/max, log scale)', fontsize=9.5)
    if i >= len(measures) - n_cols:
        ax.set_xlabel(x_label, fontsize=9.5)

handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, color=COLOR[t]) for t in COLOR]
fig.legend(handles, [LABEL[t] for t in COLOR], frameon=False, fontsize=11,
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.015))
x_title = args.volume_col if args.x_measure is None else args.x_measure
fig.suptitle(f'RT-derived behaviour measures vs. {x_title}  (* p < 0.05)',
             fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0.03, 1, 0.985])

out_name = 'lesion_volume_vs_rt_measures.png' if args.x_measure is None else f'{args.x_measure}_vs_rt_measures.png'
out_path = os.path.join(out_dir, out_name)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
log_msg(f"| FINISHED | Figure saved -> {out_path}")
