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
# Behaviour score by coarse lesion location (primary_lesion_location,   #
# from simple_stats_summary.tsv), one subplot per RT measure, paired    #
# boxes (one per task) side by side per location -- the same box-plot   #
# logic as plot_simple_stats.py section 2, generalized across measures. #
#                                                                       #
# usage: plot_measures_by_location.py --data_path /data/patrik/RT/RTM   #
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
    description='Behaviour measure by coarse lesion location, one box-plot panel per measure, '
                'paired boxes (one per task) per location.')
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
parser.add_argument('--summary_path', type=str, default=None,
                    help='Path to simple_stats_summary.tsv (default: {data_path}/simple_stats_summary.tsv)')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv (default: {data_path}/participants.tsv)')
parser.add_argument('--out_dir', type=str, default=None,
                    help='Output directory for the PNG (default: {data_path}/FIGURES)')
parser.add_argument('--measures', type=str, nargs='+', default=['original_exgaussian_tau', 'hexg_tau', 'swald_drift_inv'],
                    help='Measures to plot, one subplot each (default: original_exgaussian_tau hexg_tau swald_drift_inv)')
parser.add_argument('--min_n', type=int, default=3,
                    help='Minimum patients for a location/task to be drawn as a box rather than raw points (default: 3)')
args = parser.parse_args()

summary_path = args.summary_path or os.path.join(args.data_path, 'simple_stats_summary.tsv')
participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')
out_dir = args.out_dir or os.path.join(args.data_path, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

log_msg(f"| START | Plotting measures by location from {summary_path} + {participants_path}")


#################################
#      SHARED STYLE/COLOUR      #
#################################

TASKS = ['foreperiod', 'gonogo']
COLOR = {'foreperiod': '#2a78d6', 'gonogo': '#eb6834'}
LABEL = {'foreperiod': 'Foreperiod', 'gonogo': 'Go/No-Go'}

# the original ex-Gaussian fits predate the foreperiod_/gonogo_ naming convention
ORIGINAL_TAU_COLUMNS = {'foreperiod': 'Foreperiod_Long_tau', 'gonogo': 'GoNoGo_tau'}

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'text.color': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e5e4df', 'grid.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


#################################
#     LOAD + JOIN ON PARTICIPANT_ID #
#################################

with open(summary_path, newline='') as fh:
    summary_rows = list(csv.DictReader(fh, delimiter='\t'))
location_by_id = {r['participant_id']: r['primary_lesion_location'] for r in summary_rows}

with open(participants_path, newline='') as fh:
    part_rows = list(csv.DictReader(fh, delimiter='\t'))

def columns_for(measure):
    if measure == 'original_exgaussian_tau':
        return ORIGINAL_TAU_COLUMNS
    return {t: f'{t}_{measure}' for t in TASKS}

# per measure -> location -> task -> list of values, for patients whose PRIMARY
# (dominant-extent) lesion location is that group
vals_by_measure = {}
for measure in args.measures:
    cols = columns_for(measure)
    vals_by_loc_task = {}
    for r in part_rows:
        pid = r['participant_id']
        loc = location_by_id.get(pid)
        if not loc:
            continue
        for t in TASKS:
            raw = r.get(cols[t])
            if raw in (None, ''):
                continue
            vals_by_loc_task.setdefault(loc, {tk: [] for tk in TASKS})[t].append(float(raw))
    vals_by_measure[measure] = vals_by_loc_task
log_msg(f"| UPDATE | {len(args.measures)} measures: {args.measures}")


#################################
#        BOX PLOT GRID          #
#################################

MIN_N = args.min_n
n_measures = len(args.measures)
fig, axes = plt.subplots(n_measures, 1, figsize=(13, 5.2 * n_measures))
axes = np.atleast_1d(axes)

for ax, measure in zip(axes, args.measures):
    vals_by_loc_task = vals_by_measure[measure]
    all_locations = sorted(vals_by_loc_task.keys())
    locations = [loc for loc in all_locations
                if any(len(vals_by_loc_task[loc][t]) >= MIN_N for t in TASKS)]
    dropped = [loc for loc in all_locations if loc not in locations]
    if dropped:
        log_msg(f"| WARNING | {measure}: dropped {len(dropped)} location(s) with < {MIN_N} "
                f"patients for either task: {dropped}")

    n_loc = len(locations)
    group_width = 0.7
    box_width = group_width / len(TASKS)
    positions_by_task = {
        t: [i + (j - (len(TASKS) - 1) / 2) * box_width for i in range(n_loc)]
        for j, t in enumerate(TASKS)
    }

    for t in TASKS:
        box_data, box_pos, scatter_pos_vals = [], [], []
        for loc, pos in zip(locations, positions_by_task[t]):
            vals = vals_by_loc_task[loc][t]
            if len(vals) >= MIN_N:
                box_data.append(vals)
                box_pos.append(pos)
            elif len(vals) > 0:
                scatter_pos_vals.append((pos, vals))

        if box_data:
            ax.boxplot(box_data, positions=box_pos, widths=box_width * 0.85,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=3, markerfacecolor=COLOR[t],
                                      markeredgecolor='none', alpha=0.5),
                      medianprops=dict(color='white', linewidth=1.5),
                      boxprops=dict(facecolor=COLOR[t], edgecolor=COLOR[t], alpha=0.85),
                      whiskerprops=dict(color=COLOR[t]), capprops=dict(color=COLOR[t]))

        for pos, vals in scatter_pos_vals:
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-box_width * 0.15, box_width * 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, color=COLOR[t], s=22,
                      edgecolors='white', linewidths=0.5, zorder=5)

    tick_labels = [
        f"{loc}\n(n={', '.join(str(len(vals_by_loc_task[loc][t])) for t in TASKS)})"
        for loc in locations
    ]

    ax.set_yscale('log')
    ax.set_xticks(range(n_loc))
    ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=9)
    ax.set_xlabel('Primary lesion location  (n = Foreperiod, Go/No-Go)', fontsize=10.5)
    ax.set_ylabel('Measure value (log scale)', fontsize=10.5)
    ax.set_title(measure, fontsize=12.5, fontweight='bold')

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[t], edgecolor=COLOR[t], alpha=0.85)
                      for t in TASKS]
    ax.legend(legend_handles, [LABEL[t] for t in TASKS], frameon=False, fontsize=9.5, loc='upper right')
    ax.grid(True, axis='y', which='major', alpha=0.5)
    ax.grid(True, axis='y', which='minor', alpha=0.2)
    ax.grid(False, axis='x')

fig.suptitle('Behaviour measures by coarse lesion location', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out_path = os.path.join(out_dir, 'measures_by_lesion_location_boxplot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
log_msg(f"| FINISHED | Figure saved -> {out_path}")
