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
# Per-ROI linear-regression residual analysis of a per-ROI patient      #
# feature vs. behaviour, with a behaviour-permutation null model --     #
# independent of, and simpler than, the SBM pipeline and the            #
# ROISelection.md Approach 1/2 scripts. No SBM fitting: one feature     #
# matrix (--feature), regressed one ROI at a time against behaviour.    #
#                                                                       #
# --feature degree (default): node degree from the same binarized,     #
#   50th-percentile-of-nonzero-thresholded disconnectomes the SBM       #
#   models are fit on (via sbm/utils.py's load_graphs() -- the exact    #
#   same loader run.py/run_base.py/run_null.py use).                    #
# --feature lesion_load: raw per-ROI lesion-overlap percentage from     #
#   {atlas}_lesion_loads.tsv, continuous, no binarization/thresholding  #
#   at all -- the same source get_substrate_connectome_behaviour.py     #
#   uses for its overlap term. Verified to share node_names' exact      #
#   column order (see canon_from_lesion_loads elsewhere in this repo    #
#   for the general case; here the live data_path's lesion_loads.tsv    #
#   header already matches node_names exactly, no reindexing needed --  #
#   checked explicitly at load time, script aborts if that ever stops   #
#   holding for a different atlas/data_path).                           #
#                                                                       #
# Steps:                                                                #
#   1+2. Load the chosen feature matrix (n_patients x n_rois) and the   #
#      matching, behaviour-filtered patient list (see above).           #
#   3. Behaviour scores for the same patients, already aligned by       #
#      load_graphs (filtering is intrinsic to step 1, not a separate    #
#      join).                                                            #
#   4. Min-max rescale both the degree matrix and the behaviour vector  #
#      to [0,1], globally (one scale factor across the whole matrix/    #
#      vector, not per-ROI) -- consistent with rescale_01 usage         #
#      elsewhere in this project (e.g. run_recon_pnb.py).               #
#   5. One simple linear regression per ROI: that ROI's own [0,1]       #
#      degree column predicting the [0,1] behaviour vector. Computed    #
#      via closed-form OLS (fit_per_roi), vectorised across all 400     #
#      ROIs at once -- mathematically identical to the per-ROI          #
#      np.polyfit(x, y, 1) used in lesion_residuals.py, just fast       #
#      enough to also drive the permutation null below (~1000x this     #
#      same computation).                                               #
#   6. Per-ROI population-level residual: RMS of that ROI's per-patient #
#      residuals (NOT the raw mean, which is identically ~0 for any     #
#      intercept-fit OLS regression by construction) -- summarises how  #
#      much of the population's behavioural variance that ROI's degree  #
#      alone fails to explain. Also reported as R^2 = 1 - residual^2 /  #
#      var(behaviour_01) -- the fraction of behavioural variance that   #
#      ROI's degree alone explains.                                     #
#   6b. NULL MODEL: --n_permutations times, shuffle the patient order   #
#      of behaviour_01 (degree data untouched -- same "break the        #
#      subject-specific pairing, keep the rest fixed" logic as          #
#      functions.permute_behaviour used by run_null.py), rerun the      #
#      same 400 per-ROI regressions, and take the SINGLE HIGHEST R^2    #
#      among that permutation's own 400 ROIs (max-statistic, exactly    #
#      analogous to block_pvalues.py's null_max approach: "the best     #
#      any ROI can look by chance alone", pooled across permutations).  #
#      Every observed ROI's R^2 is then compared against this same      #
#      null_max distribution -- family-wise-error-controlled by         #
#      construction, no separate multiple-comparisons correction        #
#      needed on top.                                                   #
#   7. Map the resulting (n_rois,) residual/R^2 vectors onto the        #
#      parcellation NIfTI, one value per ROI, same convention as        #
#      lesion_residuals.py's per-ROI residual NIfTI, plus a             #
#      significant-only R^2 NIfTI (block_pvalues.py's convention).      #
#                                                                       #
# Design note: "residuals per ROI" only exists as a well-defined        #
# per-ROI quantity if each ROI gets its OWN regression (step 5) rather  #
# than one joint multivariate fit across all 400 ROIs at once (which    #
# would give per-PATIENT residuals, not per-ROI) -- that reading is     #
# what's implemented here.                                              #
#                                                                       #
# Outputs (per task):                                                   #
#   beh_regression_residuals_{score}.tsv -- per-ROI slope, intercept,   #
#     population_residual, r_squared, p_value, significant              #
#   {atlas}_beh_regression_residual_{score}.nii.gz    -- per-ROI        #
#     population residual, mapped onto the atlas                        #
#   {atlas}_beh_regression_r2_{score}.nii.gz          -- per-ROI R^2,   #
#     mapped onto the atlas                                             #
#   {atlas}_beh_regression_significant_{score}.nii.gz -- per-ROI R^2,   #
#     significant ROIs only (0 elsewhere)                               #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import sys
import csv
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sbm'))
from utils import log_msg, load_graphs


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(
    description='Per-ROI linear-regression residual analysis, with a behaviour-permutation '
                'null model: node degree (from the same binarized disconnectomes the SBM '
                'models use) regressed against behaviour, one ROI at a time, tested against '
                'chance, mapped to per-ROI NIfTIs.'
)
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau', help='Behaviour score to analyze')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--feature', type=str, default='degree', choices=['degree', 'lesion_load'],
                  help='Per-ROI predictor to regress against behaviour: "degree" (node degree '
                       'from the binarized, thresholded disconnectomes the SBM models use -- '
                       'default) or "lesion_load" (raw, continuous per-ROI lesion-overlap '
                       'percentage, no binarization)')
args.add_argument('--lesion_loads', type=str, default=None,
                  help='Path to the per-ROI lesion-load table (only used with --feature '
                       'lesion_load; default: {data_path}/{atlas}_lesion_loads.tsv)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args.add_argument('--n_permutations', type=int, default=1000,
                  help='Number of behaviour-permutation null iterations (default: 1000)')
args.add_argument('--seed', type=int, default=42,
                  help='Base random seed for the permutation null (default: 42)')
args.add_argument('--alpha', type=float, default=0.05,
                  help='Significance threshold (default: 0.05; already family-wise controlled '
                       'by the max-statistic null, no further correction applied)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# keep the original (--feature degree) output filenames unchanged for backward
# compatibility with prior runs already on disk; lesion_load gets its own suffix
# so the two feature choices never overwrite each other's outputs.
feature_suffix = '' if args.feature == 'degree' else f'_{args.feature}'

log_msg(f"| START | Per-ROI {args.feature}-vs-behaviour regression residuals | score: {args.score}")


#################################
#  1+2. LOAD FEATURE MATRIX     #
#################################

part = np.genfromtxt(os.path.join(args.data_path, 'participants.tsv'), dtype=str, delimiter='\t')
score_col = np.where(part[0] == args.score)[0][0]

atlas_meta = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[0, :].tolist()
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('label')].tolist()
n_rois = len(node_names)

if args.feature == 'degree':
    # same loader as run.py: binarizes each patient's disconnectome at their own
    # 50th-percentile-of-nonzero threshold, filters to patients with a valid score.
    discos = os.listdir(os.path.join(args.data_path, 'DISCONNECTOMES'))
    subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{args.atlas}.tsv')]

    subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects = load_graphs(
        args.data_path, args.atlas, subject_list, part, score_col)

    n_patients = len(subject_list_clean)
    log_msg(f"| UPDATE | Total subjects: {len(subject_list)}")
    log_msg(f"| UPDATE | Included: {n_patients}")
    log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
    log_msg(f"| UPDATE | Empty disconnectome: {len(empty_subjects)}")

    # adj_matrices: (n_patients, n_rois, n_rois) binary, symmetric -> row sum = node degree
    features = adj_matrices.sum(axis=2).astype(np.float64)   # (n_patients, n_rois)
    log_msg(f"| UPDATE | Node-degree matrix: {features.shape[0]} patients x {features.shape[1]} ROIs "
            f"(degree range [{features.min():.0f}, {features.max():.0f}])")

else:  # lesion_load
    lesion_loads_path = args.lesion_loads or os.path.join(args.data_path, f'{args.atlas}_lesion_loads.tsv')
    with open(lesion_loads_path, newline='') as fh:
        reader    = csv.reader(fh, delimiter='\t')
        ll_header = reader.__next__()[1:]
        ll_rows   = list(reader)

    if ll_header != node_names:
        raise SystemExit(f'{lesion_loads_path}\'s column order does not match node_names from '
                         f'{args.atlas}_areas.txt -- this script assumes an exact match (verified '
                         f'for the current data_path/atlas) and does not reindex by canonical '
                         f'name; add that reindexing before using --feature lesion_load here if '
                         f'this ever fires for a different atlas/data_path.')

    subject_list_clean     = []
    behaviour               = []
    feature_rows            = []
    subjects_missing_score  = []
    empty_subjects          = []
    for row in ll_rows:
        subject = row[0]
        val = part[part[:, 0] == subject, score_col]
        if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
            subjects_missing_score.append(subject)
            continue
        vals = np.array([float(x) for x in row[1:]], dtype=np.float64)
        if vals.sum() == 0:
            empty_subjects.append(subject)
            continue
        subject_list_clean.append(subject)
        behaviour.append(float(val[0]))
        feature_rows.append(vals)
    features = np.stack(feature_rows).astype(np.float64)   # (n_patients, n_rois), raw percentages

    n_patients = len(subject_list_clean)
    log_msg(f"| UPDATE | Total subjects (lesion_loads.tsv): {len(ll_rows)}")
    log_msg(f"| UPDATE | Included: {n_patients}")
    log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
    log_msg(f"| UPDATE | Zero lesion-overlap subjects: {len(empty_subjects)}")
    log_msg(f"| UPDATE | Lesion-load matrix: {features.shape[0]} patients x {features.shape[1]} ROIs "
            f"(range [{features.min():.2f}, {features.max():.2f}]%)")


#################################
#  3. BEHAVIOUR LABELS          #
#     (already aligned by the   #
#     filtering above)          #
#################################

behaviour = np.array(behaviour, dtype=np.float64)


#################################
#  4. RESCALE TO [0,1]          #
#################################

def rescale_01(x):
    '''global min-max rescale to [0,1]'''
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

features_01  = rescale_01(features)     # single [0,1] scale across the whole matrix
behaviour_01 = rescale_01(behaviour)    # single [0,1] scale across patients
log_msg(f"| UPDATE | Rescaled {args.feature} matrix and behaviour vector to [0,1]")


#################################
#  5+6. PER-ROI REGRESSION +    #
#        POPULATION RESIDUAL    #
#################################

def fit_per_roi(x, y):
    '''
    Closed-form OLS simple linear regression (y ~ x_j + intercept), fit
    independently per column j of x, vectorised across all columns at
    once. Mathematically identical to np.polyfit(x[:, j], y, 1) run in a
    loop, just fast enough to also drive the permutation null (called
    ~n_permutations+1 times).

    Returns (slope, intercept, population_residual, r_squared), each
    shape (n_rois,). population_residual is the RMS of the per-patient
    residuals (not the raw mean, which is ~0 by construction for any
    intercept-fit OLS regression); r_squared = 1 - population_residual^2
    / var(y). Zero-variance columns (identical feature value across every
    patient) get slope=0, intercept=mean(y), i.e. the intercept-only
    fit -- the only well-defined regression when nothing varies.
    '''
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean

    cov   = (x_centered * y_centered[:, None]).mean(axis=0)
    var_x = (x_centered ** 2).mean(axis=0)

    degenerate = var_x == 0
    slope = np.divide(cov, var_x, out=np.zeros_like(cov), where=~degenerate)
    intercept = np.where(degenerate, y_mean, y_mean - slope * x_mean)

    predicted = slope[None, :] * x + intercept[None, :]
    residual  = y[:, None] - predicted
    population_residual = np.sqrt(np.mean(residual ** 2, axis=0))

    total_var = float(y.var())
    r_squared = 1.0 - (population_residual ** 2) / total_var if total_var > 0 else np.zeros(x.shape[1])

    return slope, intercept, population_residual, r_squared, degenerate


slopes, intercepts, population_residual, r_squared, degenerate = fit_per_roi(features_01, behaviour_01)

n_degenerate = int(degenerate.sum())
if n_degenerate:
    log_msg(f"| WARNING | {n_degenerate}/{n_rois} ROIs had zero-variance {args.feature} across "
            f"included patients -- residual for these reflects behaviour's own spread only")

best_roi  = int(np.argmin(population_residual))
worst_roi = int(np.argmax(population_residual))
log_msg(f"| UPDATE | Best single-ROI predictor: {node_names[best_roi]} "
        f"(population residual = {population_residual[best_roi]:.4f}, R^2 = {r_squared[best_roi]:.4f})")
log_msg(f"| UPDATE | Worst single-ROI predictor: {node_names[worst_roi]} "
        f"(population residual = {population_residual[worst_roi]:.4f}, R^2 = {r_squared[worst_roi]:.4f})")
log_msg(f"| UPDATE | R^2 across all {n_rois} ROIs: min={r_squared.min():.4f}, "
        f"max={r_squared.max():.4f}, mean={r_squared.mean():.4f}")


#################################
#  6b. PERMUTATION NULL         #
#      (max-statistic, as in    #
#      block_pvalues.py)        #
#################################

log_msg(f"| UPDATE | Running behaviour-permutation null ({args.n_permutations} permutations)")

null_max_r2 = np.zeros(args.n_permutations)
for perm_idx in range(args.n_permutations):
    # per-permutation RNG keyed on perm_idx (not one shared RNG advanced across the
    # whole loop), matching run_null.py's convention -- reproducible independent of
    # how many permutations preceded this one.
    rng = np.random.default_rng(args.seed + perm_idx)
    perm_behaviour_01 = rng.permutation(behaviour_01)

    _, _, _, r2_perm, _ = fit_per_roi(features_01, perm_behaviour_01)
    null_max_r2[perm_idx] = r2_perm.max()

    if (perm_idx + 1) % 200 == 0:
        log_msg(f"| UPDATE | ...{perm_idx + 1}/{args.n_permutations} permutations done")

log_msg(f"| UPDATE | null_max R^2 range: [{null_max_r2.min():.4f}, {null_max_r2.max():.4f}], "
        f"mean {null_max_r2.mean():.4f}")

p_value     = np.array([(np.sum(null_max_r2 >= r_squared[j]) + 1) / (args.n_permutations + 1)
                        for j in range(n_rois)])
significant = p_value < args.alpha
n_sig       = int(significant.sum())
log_msg(f"| UPDATE | {n_sig}/{n_rois} ROIs significant at p < {args.alpha} "
        f"(family-wise controlled by the max-statistic null)")


#################################
#      SAVE PER-ROI TABLE       #
#################################

table_path = os.path.join(out_dir, f'beh_regression_residuals_{args.score}{feature_suffix}.tsv')
with open(table_path, 'w', newline='') as fh:
    writer = csv.writer(fh, delimiter='\t')
    writer.writerow(['roi_index', 'roi_name', 'slope', 'intercept', 'population_residual',
                     'r_squared', 'p_value', 'significant'])
    for j in range(n_rois):
        writer.writerow([j, node_names[j], round(float(slopes[j]), 6),
                         round(float(intercepts[j]), 6), round(float(population_residual[j]), 6),
                         round(float(r_squared[j]), 6), round(float(p_value[j]), 6),
                         bool(significant[j])])
log_msg(f"| UPDATE | Per-ROI regression table saved -> {table_path}")


#################################
#  7. MAP TO NIFTI              #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)
valid_mask     = (atlas_data > 0) & (atlas_data <= n_rois)

residual_data = np.zeros_like(atlas_data, dtype=np.float32)
residual_data[valid_mask] = population_residual[atlas_data[valid_mask] - 1]
nii_path = os.path.join(out_dir, f'{args.atlas}_beh_regression_residual_{args.score}{feature_suffix}.nii.gz')
nib.save(nib.Nifti1Image(residual_data, atlas_img.affine, atlas_img.header), nii_path)
log_msg(f"| UPDATE | Per-ROI population-residual NIfTI saved -> {nii_path}")

r2_data = np.zeros_like(atlas_data, dtype=np.float32)
r2_data[valid_mask] = r_squared[atlas_data[valid_mask] - 1]
r2_nii_path = os.path.join(out_dir, f'{args.atlas}_beh_regression_r2_{args.score}{feature_suffix}.nii.gz')
nib.save(nib.Nifti1Image(r2_data, atlas_img.affine, atlas_img.header), r2_nii_path)
log_msg(f"| UPDATE | Per-ROI R^2 NIfTI saved -> {r2_nii_path}")

sig_r2_by_roi = np.where(significant, r_squared, 0.0)
sig_data = np.zeros_like(atlas_data, dtype=np.float32)
sig_data[valid_mask] = sig_r2_by_roi[atlas_data[valid_mask] - 1]
sig_nii_path = os.path.join(out_dir, f'{args.atlas}_beh_regression_significant_{args.score}{feature_suffix}.nii.gz')
nib.save(nib.Nifti1Image(sig_data, atlas_img.affine, atlas_img.header), sig_nii_path)
log_msg(f"| UPDATE | Significant-ROI R^2 NIfTI saved ({n_sig} ROIs) -> {sig_nii_path}")

log_msg(f"| FINISHED | Behaviour-regression residual analysis saved -> {out_dir}")
