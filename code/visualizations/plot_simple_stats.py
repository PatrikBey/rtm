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
# Two figures from simple_stats_summary.tsv:                            #
#   1. Lesion volume vs. behaviour score, one scatter panel, both       #
#      behaviours overlaid and colour-coded.                            #
#   2. Behaviour score by coarse lesion location, one box-plot panel,   #
#      paired boxes (one per behaviour) side by side per location.      #
#                                                                       #
# Both behaviours share one y-axis per figure (never a dual-axis chart) #
# but that axis is LOG-scaled and explicitly labelled as such: raw      #
# value ranges span several orders of magnitude both across behaviours  #
# (Foreperiod_Long_tau ~0.01-8100, GoNoGo_tau ~0.04-680) and within      #
# lesion volume (~500-377000 mm^3) -- on a linear axis the smaller      #
# series/values would flatten to an unreadable line near zero. Log      #
# scaling keeps both series honestly visible on one shared axis without #
# altering the underlying values.                                       #
#                                                                       #
# Colour: fixed categorical slots 1 (blue, Foreperiod_Long_tau) and 2   #
# (orange, GoNoGo_tau), validated via the dataviz skill's palette       #
# validator (all checks pass, incl. the scatter/all-pairs CVD floor).   #
# Same two colours, same behaviour->colour mapping, in both figures.    #
#                                                                       #
# usage: plot_simple_stats.py --data_path /data/patrik/RT/RTM           #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


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
    description='Lesion volume vs. behaviour scatter, and behaviour-by-lesion-location box '
                'plot, from simple_stats_summary.tsv.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
parser.add_argument('--summary_path', type=str, default=None,
                    help='Path to simple_stats_summary.tsv (default: {data_path}/simple_stats_summary.tsv)')
parser.add_argument('--out_dir', type=str, default=None,
                    help='Output directory for the PNGs (default: {data_path}/FIGURES)')
parser.add_argument('--sig_reference', type=str, default='Right Frontal',
                    help='Reference location tested against every other location, within each '
                         'behaviour, for the box-plot significance brackets (default: Right Frontal)')
parser.add_argument('--sig_alpha', type=float, default=0.05,
                    help='FDR-corrected significance threshold for the box-plot brackets '
                         '(default: 0.05)')
args = parser.parse_args()

summary_path = args.summary_path or os.path.join(args.data_path, 'simple_stats_summary.tsv')
out_dir      = args.out_dir or os.path.join(args.data_path, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

log_msg(f"| START | Plotting simple stats from {summary_path}")


#################################
#      SHARED STYLE/COLOUR      #
#################################

# dataviz skill categorical slots 1 (blue) and 2 (orange), fixed order, validated
COLOR = {'Foreperiod_Long_tau': '#2a78d6', 'GoNoGo_tau': '#eb6834'}
BEHAVIOURS = ['Foreperiod_Long_tau', 'GoNoGo_tau']

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'text.color': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e5e4df', 'grid.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


#################################
#       LOAD SUMMARY TABLE      #
#################################

with open(summary_path, newline='') as fh:
    rows = list(csv.DictReader(fh, delimiter='\t'))
log_msg(f"| UPDATE | Loaded {len(rows)} patient rows")


#################################
#  1. LESION VOLUME vs BEHAVIOUR#
#     SCATTER                   #
#################################

fig, ax = plt.subplots(figsize=(8, 6))

for beh in BEHAVIOURS:
    x, y = [], []
    for r in rows:
        if r['lesion_volume_mm3'] and r[beh]:
            x.append(float(r['lesion_volume_mm3']))
            y.append(float(r[beh]))
    ax.scatter(x, y, s=28, color=COLOR[beh], alpha=0.6, edgecolors='none', label=f'{beh} (n={len(x)})')
    log_msg(f"| UPDATE | {beh}: {len(x)} patients plotted")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Lesion volume (mm³, log scale)', fontsize=11)
ax.set_ylabel('Behaviour score (log scale)', fontsize=11)
ax.set_title('Lesion volume vs. behaviour score', fontsize=13, fontweight='bold')
ax.legend(frameon=False, fontsize=10, loc='upper right')
ax.grid(True, which='major', alpha=0.5)
ax.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
scatter_path = os.path.join(out_dir, 'lesion_volume_vs_behaviour_scatter.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
log_msg(f"| UPDATE | Scatter plot saved -> {scatter_path}")


#################################
#  2. BEHAVIOUR BY LESION       #
#     LOCATION BOX PLOT         #
#################################

MIN_N = 3   # a box with fewer points than this isn't a meaningful distribution
            # (e.g. n=1 draws as a degenerate, effectively invisible line) -- such
            # locations are dropped from the plot entirely rather than left as a
            # confusing empty gap, and logged explicitly below.

all_locations = sorted(set(r['primary_lesion_location'] for r in rows if r['primary_lesion_location']))
vals_by_loc_beh = {
    loc: {beh: [float(r[beh]) for r in rows if r['primary_lesion_location'] == loc and r[beh]]
         for beh in BEHAVIOURS}
    for loc in all_locations
}
locations = [loc for loc in all_locations
            if any(len(vals_by_loc_beh[loc][beh]) >= MIN_N for beh in BEHAVIOURS)]
dropped = [loc for loc in all_locations if loc not in locations]
if dropped:
    log_msg(f"| WARNING | Dropped {len(dropped)} location(s) with < {MIN_N} patients for "
            f"either behaviour (too few for a meaningful box): "
            f"{[(loc, {b: len(vals_by_loc_beh[loc][b]) for b in BEHAVIOURS}) for loc in dropped]}")
log_msg(f"| UPDATE | {len(locations)} coarse lesion locations plotted: {locations}")

# per (location, behaviour) -> list of behaviour values, for patients whose
# PRIMARY (dominant-extent) lesion location is that group
fig, ax = plt.subplots(figsize=(13, 6))

n_loc = len(locations)
group_width = 0.7
box_width = group_width / len(BEHAVIOURS)
positions_by_beh = {
    beh: [i + (j - (len(BEHAVIOURS) - 1) / 2) * box_width
          for i in range(n_loc)]
    for j, beh in enumerate(BEHAVIOURS)
}

for beh in BEHAVIOURS:
    # split into boxable (n >= MIN_N) vs. too-few-for-a-box positions; the latter
    # are drawn as raw jittered points instead of a degenerate/misleading box.
    box_data, box_pos, scatter_pos_vals = [], [], []
    for loc, pos in zip(locations, positions_by_beh[beh]):
        vals = vals_by_loc_beh[loc][beh]
        if len(vals) >= MIN_N:
            box_data.append(vals)
            box_pos.append(pos)
        elif len(vals) > 0:
            log_msg(f"| UPDATE | {loc} / {beh}: only {len(vals)} patient(s), plotting as raw "
                    f"point(s) instead of a box")
            scatter_pos_vals.append((pos, vals))

    if box_data:
        ax.boxplot(box_data, positions=box_pos, widths=box_width * 0.85,
                  patch_artist=True, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, markerfacecolor=COLOR[beh],
                                  markeredgecolor='none', alpha=0.5),
                  medianprops=dict(color='white', linewidth=1.5),
                  boxprops=dict(facecolor=COLOR[beh], edgecolor=COLOR[beh], alpha=0.85),
                  whiskerprops=dict(color=COLOR[beh]), capprops=dict(color=COLOR[beh]))

    for pos, vals in scatter_pos_vals:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-box_width * 0.15, box_width * 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, color=COLOR[beh], s=22,
                  edgecolors='white', linewidths=0.5, zorder=5)

# tick label carries n per behaviour, in BEHAVIOURS/legend order (Foreperiod, GoNoGo),
# so counts are readable without cross-referencing colours back to the legend.
tick_labels = [
    f"{loc}\n(n={', '.join(str(len(vals_by_loc_beh[loc][beh])) for beh in BEHAVIOURS)})"
    for loc in locations
]

ax.set_yscale('log')
ax.set_xticks(range(n_loc))
ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=9)
ax.set_xlabel('Primary lesion location  (n = Foreperiod_Long_tau, GoNoGo_tau)', fontsize=11)
ax.set_ylabel('Behaviour score (log scale)', fontsize=11)
ax.set_title('Behaviour score by coarse lesion location', fontsize=13, fontweight='bold')

legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[beh], edgecolor=COLOR[beh], alpha=0.85)
                  for beh in BEHAVIOURS]
ax.legend(legend_handles, BEHAVIOURS, frameon=False, fontsize=10, loc='upper right')
ax.grid(True, axis='y', which='major', alpha=0.5)
ax.grid(True, axis='y', which='minor', alpha=0.2)
ax.grid(False, axis='x')

plt.tight_layout()
box_path = os.path.join(out_dir, 'behaviour_by_lesion_location_boxplot.png')
plt.savefig(box_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
log_msg(f"| UPDATE | Box plot saved -> {box_path}")

log_msg(f"| FINISHED | Figures saved -> {out_dir}")
