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
# ANOVA/ANCOVA analysis of the two real behaviour scores (Foreperiod_   #
# Long_tau, GoNoGo_tau) as a function of coarse lesion location, lesion #
# volume, and both together, using simple_stats_summary.tsv.            #
#                                                                       #
# Location = primary_lesion_location (the single dominant coarse        #
# region per patient) -- not the full multi-region lesion_location      #
# list, since ANOVA needs one categorical factor per observation, and   #
# this matches the factor already used in plot_simple_stats.py's box   #
# plot.                                                                 #
#                                                                       #
# Both behaviour scores and lesion volume are strongly right-skewed on  #
# their raw scale (skew 7.8 / 2.1 / 1.2 respectively; Shapiro-Wilk      #
# p<1e-11 for all three) -- ANOVA assumes roughly normal, homoscedastic #
# residuals, which raw-scale values badly violate. All three variables  #
# are therefore natural-log-transformed before fitting (skew drops to   #
# ~1.7 / 0.6 / 1.0), consistent with the log scaling already used for   #
# these same variables in plot_simple_stats.py. NOTE: log-transforming  #
# does not fully normalize either behaviour score -- both show a        #
# genuine bimodal split (a cluster under ~1 and a cluster above ~6,     #
# almost nothing between), most likely a units/measurement-convention   #
# inconsistency in participants.tsv rather than a real bimodal effect   #
# (flagged when simple_stats_summary.tsv was first built) -- the        #
# p-values below should be read with that residual non-normality in    #
# mind, not as a fully assumption-clean result.                         #
#                                                                       #
# Locations with fewer than --min_n patients for a given behaviour are  #
# dropped before fitting that behaviour's location-based models (a      #
# single-observation group contributes no within-group variance and     #
# can make the design rank-deficient) -- same threshold and rationale   #
# as plot_simple_stats.py's box plot.                                   #
#                                                                       #
# Three models per behaviour, each summarised via a Type II ANOVA table #
# (appropriate for this unbalanced, no-interaction design; Type III is   #
# only needed once an interaction term is added, Type I is order-        #
# dependent):                                                            #
#   1. log(behaviour) ~ C(location)                    -- location alone #
#   2. log(behaviour) ~ log(lesion_volume_mm3)          -- volume alone   #
#   3. log(behaviour) ~ C(location) + log(lesion_volume_mm3)  -- both     #
#      (ANCOVA, additive -- no location x volume interaction term)       #
#                                                                       #
# usage: anova_location_volume.py --data_path /data/patrik/RT/RTM       #
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
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import skew, shapiro


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
    description='ANOVA/ANCOVA of behaviour scores by coarse lesion location, lesion volume, '
                'and both together, using simple_stats_summary.tsv.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
parser.add_argument('--summary_path', type=str, default=None,
                    help='Path to simple_stats_summary.tsv, source of primary_lesion_location '
                         'and lesion_volume_mm3 (default: {data_path}/simple_stats_summary.tsv)')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv, source of the --behaviours columns -- joined '
                         'to summary_path on participant_id (default: {data_path}/participants.tsv)')
parser.add_argument('--behaviours', type=str, nargs='+', default=['Foreperiod_Long_tau', 'GoNoGo_tau'],
                    help='participants.tsv column(s) to run the ANOVA/ANCOVA on '
                         '(default: Foreperiod_Long_tau GoNoGo_tau)')
parser.add_argument('--location_col', type=str, default='primary_lesion_location',
                    help='Categorical grouping column to use as the location factor '
                         '(default: primary_lesion_location, from summary_path)')
parser.add_argument('--location_source', type=str, choices=['summary', 'participants'], default='summary',
                    help='Which table --location_col lives in (default: summary, i.e. summary_path; '
                         '\'participants\' reads it from participants_path instead, e.g. Group_label)')
parser.add_argument('--exclude_locations', type=str, nargs='+', default=[],
                    help='Category values to drop entirely before fitting (e.g. an '
                         '"Unclassified" catch-all group that is not a real location)')
parser.add_argument('--min_n', type=int, default=3,
                    help='Drop locations with fewer than this many patients for a given '
                         'behaviour before fitting that behaviour\'s location models (default: 3)')
parser.add_argument('--out_path', type=str, default=None,
                    help='Output TSV path (default: {data_path}/anova_location_volume_summary.tsv)')
args = parser.parse_args()

summary_path = args.summary_path or os.path.join(args.data_path, 'simple_stats_summary.tsv')
participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')
out_path     = args.out_path or os.path.join(args.data_path, 'anova_location_volume_summary.tsv')

BEHAVIOURS = args.behaviours

log_msg(f"| START | ANOVA/ANCOVA of location + lesion volume vs. behaviour | {summary_path}")


#################################
#      LOAD + TRANSFORM         #
#################################

LOC_COL = args.location_col

# volume + behaviour columns come from participants.tsv; the location factor comes from
# either table (--location_source), joined on participant_id so any participants.tsv
# column can be analysed/grouped-by, not just the two/one this script originally shipped
# with (primary_lesion_location from simple_stats_summary.tsv, log Foreperiod/GoNoGo tau).
participants_df = pd.read_csv(participants_path, sep='\t', dtype=str)
part_cols = ['participant_id', 'lesion_volume_mm3'] + BEHAVIOURS
if args.location_source == 'participants':
    part_cols = part_cols + [LOC_COL]
participants_df = participants_df[part_cols]

if args.location_source == 'summary':
    summary_df = pd.read_csv(summary_path, sep='\t', dtype=str)[['participant_id', LOC_COL]]
    df_all = summary_df.merge(participants_df, on='participant_id', how='left')
else:
    df_all = participants_df

if args.exclude_locations:
    before = len(df_all)
    df_all = df_all[~df_all[LOC_COL].isin(args.exclude_locations)]
    log_msg(f"| UPDATE | Excluded {before - len(df_all)} patient(s) in {args.exclude_locations}")

df_all['log_lesion_volume'] = np.log(df_all['lesion_volume_mm3'].astype(float))

for col, label in [(b, b) for b in BEHAVIOURS] + [('lesion_volume_mm3', 'lesion_volume_mm3')]:
    vals = df_all[col].replace('', np.nan).dropna().astype(float).values
    sample = vals if len(vals) <= 5000 else vals[:5000]   # shapiro caps out around 5000
    log_msg(f"| UPDATE | {label}: raw skew={skew(vals):.2f}, raw Shapiro p={shapiro(sample).pvalue:.2e}, "
            f"log skew={skew(np.log(vals)):.2f}, log Shapiro p={shapiro(np.log(sample)).pvalue:.2e}")


#################################
#      PER-BEHAVIOUR ANOVA      #
#################################

results_rows = []

for beh in BEHAVIOURS:
    log_msg(f"##### {beh} #####")

    df = df_all[[beh, 'lesion_volume_mm3', 'log_lesion_volume', LOC_COL]].copy()
    df[beh] = df[beh].replace('', np.nan)
    df = df.dropna(subset=[beh])
    df[beh] = df[beh].astype(float)
    df[f'log_{beh}'] = np.log(df[beh])

    # drop locations with too few patients for THIS behaviour, and rows with no location
    # (needed for models 1 and 3, not for model 2)
    df_loc = df.dropna(subset=[LOC_COL])
    df_loc = df_loc[df_loc[LOC_COL] != '']
    loc_counts = df_loc[LOC_COL].value_counts()
    keep_locs  = loc_counts[loc_counts >= args.min_n].index
    dropped    = sorted(set(loc_counts.index) - set(keep_locs))
    if dropped:
        log_msg(f"| WARNING | Dropping {len(dropped)} location(s) with < {args.min_n} patients: "
                f"{[(l, int(loc_counts[l])) for l in dropped]}")
    df_loc = df_loc[df_loc[LOC_COL].isin(keep_locs)]
    log_msg(f"| UPDATE | Model 1/3 (location): n={len(df_loc)} patients across {len(keep_locs)} locations")
    log_msg(f"| UPDATE | Model 2 (volume alone): n={len(df)} patients")

    models = {
        'location_only':        (f'log_{beh} ~ C({LOC_COL})', df_loc),
        'volume_only':           (f'log_{beh} ~ log_lesion_volume', df),
        'location_plus_volume':  (f'log_{beh} ~ C({LOC_COL}) + log_lesion_volume', df_loc),
    }

    for model_name, (formula, data) in models.items():
        fit = smf.ols(formula, data=data).fit()
        table = anova_lm(fit, typ=2)
        log_msg(f"--- {beh} | {model_name} | formula: {formula} | n={int(fit.nobs)} | "
                f"R2={fit.rsquared:.4f}, adj.R2={fit.rsquared_adj:.4f} ---")
        for term, row in table.iterrows():
            if term == 'Residual':
                continue
            log_msg(f"    {term:35s} df={row['df']:.0f}  F={row['F']:.3f}  p={row['PR(>F)']:.4g}")
            results_rows.append({
                'behaviour': beh, 'model': model_name, 'term': term,
                'df': row['df'], 'F': round(float(row['F']), 4), 'p_value': row['PR(>F)'],
                'model_r2': round(float(fit.rsquared), 4), 'model_adj_r2': round(float(fit.rsquared_adj), 4),
                'n': int(fit.nobs),
            })


#################################
#      SAVE SUMMARY TABLE       #
#################################

with open(out_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['behaviour', 'model', 'term', 'df', 'F', 'p_value',
                                            'model_r2', 'model_adj_r2', 'n'], delimiter='\t')
    writer.writeheader()
    writer.writerows(results_rows)

log_msg(f"| UPDATE | ANOVA summary table saved -> {out_path}")
log_msg(f"| FINISHED | ANOVA/ANCOVA analysis complete")
