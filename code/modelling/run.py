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
# The following script derives robust single-parameter reaction-time    #
# descriptors per patient, for use as behaviour scores in downstream    #
# SBM modelling, from a raw per-trial RT table.                         #
#                                                                       #
# It replaces per-subject maximum-likelihood ex-Gaussian tau, which at  #
# ~18 trials per subject has a split-half reliability of only 0.47 and  #
# a bimodal distribution driven by two distinct estimation failures:    #
# collapse to the tau -> 0 boundary in negatively-skewed samples (39 of #
# 39 such subjects), and unbounded inflation from single contaminated   #
# trials (max/median up to 80). The aim throughout is to KEEP the right #
# tail -- which is the hypothesised marker of disease severity -- while #
# bounding the influence any one trial can have on the estimate.        #
#                                                                       #
# Three estimators are produced side by side:                           #
#                                                                       #
#   1. Shifted Wald (ML, per subject). Two shape parameters plus a      #
#      non-decision shift; skew is structural. The drift rate reads as  #
#      evidence-accumulation efficiency.                                #
#                                                                       #
#   2. Shifted lognormal (ML, per subject, 1-D profile over the shift). #
#      Same motivation, multiplicative rather than additive tail.       #
#                                                                       #
#   3. Hierarchical ex-Gaussian with an explicit lapse component        #
#      (Bayesian, all subjects jointly). Keeps the tau parameterisation #
#      that originally motivated this analysis, but partial pooling on  #
#      log tau plus a flat lapse mixture component remove both failure  #
#      modes without trimming the tail.                                 #
#                                                                       #
#   4. Bounded tail-mass scores: the proportion of a subject's trials   #
#      exceeding the pooled healthy-control 90th/95th percentile. These #
#      had the best split-half reliability of any tail measure tested   #
#      (0.86 / 0.83) and cannot blow up by construction.                #
#                                                                       #
# Estimators 1-2 have NO lapse component and so remain sensitive to     #
# contaminated trials; they are the confirmatory arm. Estimator 3 and   #
# the tail-mass scores are the robust arm. Agreement between the two    #
# arms is the check worth making before using any of them downstream.   #
#                                                                       #
# usage: run.py --rt_path /path/to/RT_FPtrials.csv --condition long     #
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

from utils import log_msg
from utils import load_rt_trials, load_rt_trials_sectioned, select_condition, clean_rts, pooled_reference_quantile
from utils import frac_above, fit_shifted_wald, fit_shifted_lognormal, residualise_on_speed
from utils import exgaussian_logpdf, build_hierarchical_exgaussian_lapse


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Robust single-parameter RT modelling for SBM behaviour scores.')
args.add_argument('--rt_path', type=str, required=True,
                  help='Path to the raw per-trial RT csv (wide format, one row per subject)')
args.add_argument('--out_path', type=str, default=None,
                  help='Output TSV path (default: {dirname(rt_path)}/rt_model_parameters.tsv)')
args.add_argument('--layout', type=str, default='plain', choices=['plain', 'sectioned'],
                  help='Trial-file layout. "plain": one flat run of trials whose condition is '
                       'given per column (the foreperiod export). "sectioned": condition blocks, '
                       'with the RT blocks identified by --section_marker (the Go/No-Go export)')
args.add_argument('--section_marker', type=str, default='Speed',
                  help='For --layout sectioned: section header marking the start of the '
                       'reaction-time blocks (default: Speed)')
args.add_argument('--condition', type=str, default='long',
                  help='Trial condition label to model, or "all" to pool conditions (default: long)')
args.add_argument('--scale_factor', type=float, default=10.0,
                  help='Divide raw trial values by this to obtain milliseconds. The raw file is '
                       'a factor ~10 above plausible RTs on four independent diagnostics; set to '
                       '1.0 if the recording units are confirmed to be ms already (default: 10.0)')
args.add_argument('--rt_min', type=float, default=100.0,
                  help='Absolute lower cleaning bound in ms, removing anticipations (default: 100)')
args.add_argument('--rt_max', type=float, default=np.inf,
                  help='Absolute upper cleaning bound in ms. Deliberately infinite by default: '
                       'the right tail is the effect of interest and the lapse component, not '
                       'trimming, is what handles contamination (default: inf)')
args.add_argument('--min_trials', type=int, default=12,
                  help='Subjects with fewer surviving trials are reported as missing rather than '
                       'fitted (default: 12)')
args.add_argument('--control_prefix', type=str, default='HC',
                  help='Subject-id prefix identifying the healthy-control reference group used '
                       'for the tail-mass thresholds (default: HC)')
args.add_argument('--id_prefix', type=str, default='',
                  help='String prepended to each subject id in the output, to match the id '
                       'convention of participants.tsv (e.g. "sub-") (default: empty)')
args.add_argument('--draws', type=int, default=1000,
                  help='Posterior draws per chain for the hierarchical model (default: 1000)')
args.add_argument('--tune', type=int, default=1000,
                  help='Tuning steps per chain for the hierarchical model (default: 1000)')
args.add_argument('--chains', type=int, default=4,
                  help='Number of MCMC chains for the hierarchical model (default: 4)')
args.add_argument('--target_accept', type=float, default=0.95,
                  help='NUTS target acceptance rate; raise toward 0.99 if divergences appear '
                       '(default: 0.95)')
args.add_argument('--max_treedepth', type=int, default=12,
                  help='NUTS maximum tree depth. Saturating it costs efficiency rather than '
                       'correctness, but persistent saturation signals hard geometry (default: 12)')
args.add_argument('--idata_path', type=str, default=None,
                  help='Optionally dump the FULL posterior (every draw of every parameter, '
                       'tens of MB) to this NetCDF path. Off by default: the per-subject '
                       'parameters are already in the output TSV, as are per-subject r_hat and '
                       'ESS, so the full trace is only needed for deeper diagnostics')
args.add_argument('--seed', type=int, default=42,
                  help='Random seed for the sampler (default: 42)')
args = args.parse_args()

out_path = args.out_path or os.path.join(os.path.dirname(os.path.abspath(args.rt_path)),
                                         'rt_model_parameters.tsv')

log_msg(f"| START | Robust single-parameter RT modelling")
log_msg(f"| UPDATE | Trial file: {args.rt_path}")
log_msg(f"| UPDATE | Condition: {args.condition} | scale factor: {args.scale_factor}")


#################################
#         LOAD TRIALS           #
#################################

if args.layout == 'sectioned':
    subjects, trial_labels, rt_by_subject = load_rt_trials_sectioned(
        args.rt_path, section_marker=args.section_marker, scale_factor=args.scale_factor)
else:
    subjects, trial_labels, rt_by_subject = load_rt_trials(args.rt_path, scale_factor=args.scale_factor)
log_msg(f"| UPDATE | Loaded {len(subjects)} subjects | trial labels present: "
        f"{sorted(set(l for l in trial_labels if l))}")

rt_raw = select_condition(subjects, rt_by_subject, args.condition)
rt_clean = {s: clean_rts(v, args.rt_min, args.rt_max) for s, v in rt_raw.items()}

fitted = [s for s in subjects if rt_clean[s].size >= args.min_trials]
skipped = [s for s in subjects if s not in fitted]
log_msg(f"| UPDATE | {len(fitted)} subjects with >= {args.min_trials} usable trials "
        f"({len(skipped)} reported as missing)")
log_msg(f"| UPDATE | Trials per fitted subject: min={min(rt_clean[s].size for s in fitted)} "
        f"median={int(np.median([rt_clean[s].size for s in fitted]))} "
        f"max={max(rt_clean[s].size for s in fitted)}")


#################################
#   TAIL-MASS SCORES (AV. 2)    #
#################################

reference = {s: rt_clean[s] for s in fitted}
hc_p90 = pooled_reference_quantile(reference, 90, args.control_prefix)
hc_p95 = pooled_reference_quantile(reference, 95, args.control_prefix)
n_controls = sum(1 for s in fitted if s.startswith(args.control_prefix))
log_msg(f"| UPDATE | Control reference ({n_controls} subjects, prefix '{args.control_prefix}'): "
        f"P90={hc_p90:.1f} ms | P95={hc_p95:.1f} ms")

tail_p90 = {s: frac_above(rt_clean[s], hc_p90) for s in fitted}
tail_p95 = {s: frac_above(rt_clean[s], hc_p95) for s in fitted}

# general slowing (median RT) and tail-SPECIFIC slowing (tail mass with the
# speed effect regressed out), reported alongside each other so the two can be
# compared downstream rather than conflated
median_rt = {s: float(np.median(rt_clean[s])) for s in fitted}
resid_vals, resid_slope, resid_int, resid_r2 = residualise_on_speed(
    [tail_p90[s] for s in fitted], [median_rt[s] for s in fitted])
tail_p90_resid = dict(zip(fitted, resid_vals))
log_msg(f"| UPDATE | Tail mass vs. general slowing: tail_p90 = {resid_slope:.4f}*log(median RT) "
        f"+ {resid_int:.4f}, R2={resid_r2:.3f} -- i.e. {resid_r2*100:.0f}% of the tail score is "
        f"explained by overall speed; the residual column isolates the rest")


#################################
#  SHIFTED 2-PARAM FITS (AV. 5) #
#################################

wald = {}
lognorm = {}
for subject in fitted:
    wald[subject] = fit_shifted_wald(rt_clean[subject])
    lognorm[subject] = fit_shifted_lognormal(rt_clean[subject])

n_wald_ok = sum(1 for s in fitted if wald[s][3])
n_lognorm_ok = sum(1 for s in fitted if lognorm[s][3])
log_msg(f"| UPDATE | Shifted Wald converged for {n_wald_ok}/{len(fitted)} subjects")
log_msg(f"| UPDATE | Shifted lognormal converged for {n_lognorm_ok}/{len(fitted)} subjects")


#################################
# HIERARCHICAL EX-GAUSSIAN (A1) #
#################################

import pymc as pm

subject_position = {s: i for i, s in enumerate(fitted)}
rt_concat = np.concatenate([rt_clean[s] for s in fitted])
subject_index = np.concatenate([np.full(rt_clean[s].size, subject_position[s], dtype=np.int64)
                                for s in fitted])
lapse_max = float(rt_concat.max()) * 1.01
log_msg(f"| UPDATE | Hierarchical model: {rt_concat.size} trials, {len(fitted)} subjects, "
        f"lapse component Uniform(0, {lapse_max:.0f})")

model = build_hierarchical_exgaussian_lapse(rt_concat, subject_index, len(fitted), lapse_max)
with model:
    idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                      target_accept=args.target_accept, random_seed=args.seed,
                      nuts=dict(max_treedepth=args.max_treedepth),
                      progressbar=False)

hexg_mu    = idata.posterior['mu_i'].mean(dim=('chain', 'draw')).values
hexg_sigma = idata.posterior['sigma_i'].mean(dim=('chain', 'draw')).values
hexg_tau   = idata.posterior['tau_i'].mean(dim=('chain', 'draw')).values
hexg_total = idata.posterior['total_i'].mean(dim=('chain', 'draw')).values
hexg_tail  = idata.posterior['tail_i'].mean(dim=('chain', 'draw')).values
hexg_w     = np.full(len(fitted), float(idata.posterior['w'].mean()))

# the deliverable is the per-subject parameters, which go to the TSV below;
# the full posterior is tens of MB and only needed for deeper diagnostics, so
# it is written only on explicit request. Both this and the diagnostics are
# non-fatal: sampling costs hours and nothing after it may abort the run
# before the results table is written.
if args.idata_path:
    try:
        idata.to_netcdf(args.idata_path)
        log_msg(f"| UPDATE | Full posterior saved -> {args.idata_path}")
    except Exception as exc:
        log_msg(f"| WARNING | Could not save posterior ({type(exc).__name__}: {exc}). Continuing.")

# per-subject convergence, carried as columns in the output table so a bad
# subject is visible in the same row as its parameters
rhat_by_subject = {s: np.nan for s in fitted}
ess_by_subject = {s: np.nan for s in fitted}
try:
    import arviz as az
    summary = az.summary(idata, var_names=['total_i', 'tail_i'], kind='diagnostics')
    rhat = summary['r_hat'].values.reshape(2, -1)
    ess = summary['ess_bulk'].values.reshape(2, -1)
    for s in fitted:
        rhat_by_subject[s] = float(np.max(rhat[:, subject_position[s]]))
        ess_by_subject[s] = float(np.min(ess[:, subject_position[s]]))
    worst_rhat = float(np.nanmax(list(rhat_by_subject.values())))
    min_ess = float(np.nanmin(list(ess_by_subject.values())))
    n_bad = int(np.sum(np.array(list(rhat_by_subject.values())) > 1.01))
    log_msg(f"| UPDATE | Convergence | worst r_hat={worst_rhat:.4f} ({n_bad}/{len(fitted)} "
            f"subjects above 1.01) | min bulk ESS={min_ess:.0f}")
    if worst_rhat > 1.01 or min_ess < 400:
        log_msg(f"| WARNING | Hierarchical estimates have NOT converged to a usable standard. "
                f"Treat hexg_* columns as provisional; the per-subject hexg_rhat / hexg_ess "
                f"columns show which subjects are affected.")
except Exception as exc:
    log_msg(f"| WARNING | Convergence diagnostics failed ({type(exc).__name__}: {exc}). Continuing.")

n_divergent = int(idata.sample_stats['diverging'].values.sum())
log_msg(f"| UPDATE | Sampling finished | divergences: {n_divergent} "
        f"(raise --target_accept toward 0.99 if this is non-trivial)")
log_msg(f"| UPDATE | Subject mean (total_i) spread: {hexg_total.min():.0f}-{hexg_total.max():.0f} ms "
        f"(must track the empirical between-subject spread, not collapse to the group mean)")
log_msg(f"| UPDATE | Tail fraction spread: {hexg_tail.min():.3f}-{hexg_tail.max():.3f} | "
        f"group lapse rate={float(idata.posterior['w'].mean()):.4f}")

# per-subject mean lapse responsibility: the posterior share of that subject's
# trials the model attributes to the flat lapse component rather than to the
# ex-Gaussian. High values flag subjects whose tau would previously have been
# driven by contamination.
lapse_resp = {}
for subject in fitted:
    i = subject_position[subject]
    rt = rt_clean[subject]
    dens_exg = np.exp(exgaussian_logpdf(rt, hexg_mu[i], hexg_sigma[i], hexg_tau[i]))
    dens_lap = 1.0 / lapse_max
    numerator = hexg_w[i] * dens_lap
    lapse_resp[subject] = float(np.mean(numerator / (numerator + (1.0 - hexg_w[i]) * dens_exg)))


#################################
#      ASSEMBLE + SAVE TABLE    #
#################################

fieldnames = ['participant_id', 'n_trials_total', 'n_trials_used',
              'swald_drift', 'swald_boundary', 'swald_shift', 'swald_converged',
              'slognorm_shift', 'slognorm_mu', 'slognorm_sigma', 'slognorm_converged',
              'hexg_total', 'hexg_tail_fraction', 'hexg_mu', 'hexg_sigma', 'hexg_tau',
              'hexg_log_tau', 'hexg_lapse_rate', 'hexg_lapse_responsibility',
              'hexg_rhat', 'hexg_ess_bulk',
              'median_rt', 'tail_frac_above_hc_p90', 'tail_frac_above_hc_p95',
              'tail_frac_above_hc_p90_speed_residual']

with open(out_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for subject in subjects:
        row = {k: '' for k in fieldnames}
        row['participant_id'] = f'{args.id_prefix}{subject}'
        row['n_trials_total'] = rt_raw[subject].size
        row['n_trials_used'] = rt_clean[subject].size
        if subject in subject_position:
            i = subject_position[subject]
            drift, boundary, shift, wald_ok = wald[subject]
            ln_shift, ln_mu, ln_sigma, ln_ok = lognorm[subject]
            row['swald_drift']    = round(drift, 6)
            row['swald_boundary'] = round(boundary, 4)
            row['swald_shift']    = round(shift, 4)
            row['swald_converged'] = int(wald_ok)
            row['slognorm_shift'] = round(ln_shift, 4)
            row['slognorm_mu']    = round(ln_mu, 6)
            row['slognorm_sigma'] = round(ln_sigma, 6)
            row['slognorm_converged'] = int(ln_ok)
            row['hexg_total']         = round(float(hexg_total[i]), 4)
            row['hexg_tail_fraction'] = round(float(hexg_tail[i]), 6)
            row['hexg_mu']      = round(float(hexg_mu[i]), 4)
            row['hexg_sigma']   = round(float(hexg_sigma[i]), 4)
            row['hexg_tau']     = round(float(hexg_tau[i]), 4)
            row['hexg_log_tau'] = round(float(np.log(hexg_tau[i])), 6)
            row['hexg_lapse_rate'] = round(float(hexg_w[i]), 6)
            row['hexg_lapse_responsibility'] = round(lapse_resp[subject], 6)
            row['hexg_rhat']      = round(rhat_by_subject[subject], 4)
            row['hexg_ess_bulk']  = round(ess_by_subject[subject], 1)
            row['median_rt'] = round(median_rt[subject], 3)
            row['tail_frac_above_hc_p90'] = round(tail_p90[subject], 6)
            row['tail_frac_above_hc_p90_speed_residual'] = round(tail_p90_resid[subject], 6)
            row['tail_frac_above_hc_p95'] = round(tail_p95[subject], 6)
        writer.writerow(row)

log_msg(f"| UPDATE | {len(subjects)} rows written ({len(fitted)} fitted, {len(skipped)} missing) "
        f"-> {out_path}")
log_msg(f"| FINISHED | RT model parameters saved -> {out_path}")
