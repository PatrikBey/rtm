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
# Level-1 counterpart of roi_pvalues.py: implements Approach 1 from     #
# doc/ROISelection.md (null-referenced significance per ROI) using the  #
# level-1 SBM partition instead of level 0. Level 1 has fewer, larger   #
# blocks than level 0, so the ROI-level pseudo-replication (every ROI   #
# in a block shares one z-score) is spread across fewer distinct        #
# values, which is a different bias-variance trade-off than level 0 -   #
# see block_pvalues.py for a version that removes the pseudo-           #
# replication entirely by testing at the block level.                   #
#                                                                       #
# The null z-scores are read directly from each permutation's           #
# roi_block_assignments_{task}.csv (zscore_1 column) rather than from   #
# a pre-built summary table, since the existing permutation_zscores.tsv #
# is level-0-specific.                                                  #
#                                                                       #
# Outputs (mirrors roi_pvalues.py):                                     #
#   roi_pvalues_lvl1.tsv                                                #
#   {atlas}_lvl1_significant_{task}.nii.gz                              #
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
from utils import log_msg


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Level-1 ROI-specific null-referenced p-values (ROISelection.md Approach 1).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+',
                  default=['Foreperiod_Long_tau', 'GoNoGo_tau', 'SATO_Accuracy_tau'],
                  help='Behaviour scores to evaluate (default: all three available tasks)')
args.add_argument('--level', type=int, default=1, help='Hierarchy level to test (default: 1)')
args.add_argument('--alpha', type=float, default=0.05, help='FDR-corrected significance threshold (default: 0.05)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMFITTING run directory (default: _singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | ROI p-values lvl{args.level} (Approach 1) | tasks: {args.tasks}")


#################################
#     BENJAMINI-HOCHBERG FDR    #
#################################

def bh_fdr(pvals):
    '''Benjamini-Hochberg FDR-corrected p-values.'''
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.clip(ranked, 0, 1)
    return corrected


#################################
#      LOAD ATLAS TEMPLATE      #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)


#################################
#     PER-TASK COMPUTATION      #
#################################

task_results = {}   # task -> {roi_name: (observed_z, p_raw, p_fdr)}

for task in args.tasks:
    fit_dir  = os.path.join(args.data_path, 'SBMFITTING', f'SBM_{args.atlas}_{task}{args.fit_suffix}')
    null_dir = os.path.join(args.data_path, 'SBMNULL', f'SBM_{args.atlas}_{task}_NULL')

    # ---- observed lvl-{level} ROI z-score: block id per ROI x block zscore ---- #
    roi_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(roi_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    roi_index  = np.array([int(r['roi_index']) for r in rows])
    roi_name   = np.array([r['roi_name'] for r in rows])
    roi_block  = np.array([int(r[f'level_{args.level}']) for r in rows])

    block_path = os.path.join(fit_dir, f'SBM_block_scores_lvl{args.level}_{task}.csv')
    with open(block_path, newline='') as fh:
        block_rows = list(csv.DictReader(fh))
    block_zscore = {int(r['block']): float(r['zscore']) for r in block_rows}

    order       = np.argsort(roi_index)
    roi_index   = roi_index[order]
    roi_name    = roi_name[order]
    observed_z  = np.array([block_zscore[b] for b in roi_block[order]])

    # ---- null distribution: roi x permutation lvl-{level} z-scores, read   ---- #
    # ---- directly from each perm_XXXXX/roi_block_assignments_{task}.csv    ---- #
    perm_dirs = sorted(d for d in os.listdir(null_dir) if d.startswith('perm_') and
                       os.path.isdir(os.path.join(null_dir, d)))
    zscore_col = f'zscore_{args.level}'
    null_roi   = None
    null_cols  = []
    for perm in perm_dirs:
        perm_csv = os.path.join(null_dir, perm, f'roi_block_assignments_{task}.csv')
        with open(perm_csv, newline='') as fh:
            perm_rows = list(csv.DictReader(fh))
        names   = [r['roi_name'] for r in perm_rows]
        zscores = [float(r[zscore_col]) for r in perm_rows]
        if null_roi is None:
            null_roi = names
        elif names != null_roi:
            raise ValueError(f"[{task}] ROI order in {perm_csv} does not match the first permutation")
        null_cols.append(zscores)
    null_roi = np.array(null_roi)
    null_z   = np.array(null_cols, dtype=np.float64).T   # roi x n_perm
    n_perm   = null_z.shape[1]

    if not np.array_equal(roi_name, null_roi):
        raise ValueError(f"[{task}] ROI order differs between observed fit and null permutations")

    log_msg(f"| UPDATE | {task}: {n_perm} permutations loaded for level {args.level}")

    # ---- one-sided empirical p-value per ROI, then BH-FDR across ROIs ---- #
    p_raw = np.array([(np.sum(null_z[i] >= observed_z[i]) + 1) / (n_perm + 1)
                      for i in range(len(observed_z))])
    p_fdr = bh_fdr(p_raw)

    task_results[task] = {name: (observed_z[i], p_raw[i], p_fdr[i])
                          for i, name in enumerate(roi_name)}

    n_sig = int(np.sum(p_fdr < args.alpha))
    log_msg(f"| UPDATE | {task}: {n_sig}/{len(roi_name)} ROIs significant at FDR < {args.alpha}")

    # ---- significant-ROI NIfTI for this task ---- #
    sig_z = np.where(p_fdr < args.alpha, observed_z, 0.0)
    sig_by_index = np.zeros(int(roi_index.max()) + 1)
    sig_by_index[roi_index] = sig_z

    sig_data  = np.zeros_like(atlas_data, dtype=np.float32)
    valid_mask = (atlas_data > 0) & (atlas_data <= len(sig_by_index))
    sig_data[valid_mask] = sig_by_index[atlas_data[valid_mask] - 1]

    sig_path = os.path.join(out_dir, f'{args.atlas}_lvl{args.level}_significant_{task}.nii.gz')
    nib.save(nib.Nifti1Image(sig_data, atlas_img.affine, atlas_img.header), sig_path)
    log_msg(f"| UPDATE | {task}: significant-ROI NIfTI saved → {sig_path}")


#################################
#      ROI x TASK P-VALUES      #
#################################

roi_names_ref = None
for task, res in task_results.items():
    names = sorted(res.keys())
    if roi_names_ref is None:
        roi_names_ref = names
    elif names != roi_names_ref:
        raise ValueError("ROI names differ across tasks - cannot build a shared ROI x task table")

table_path = os.path.join(out_dir, f'roi_pvalues_lvl{args.level}.tsv')
with open(table_path, 'w', newline='') as fh:
    writer = csv.writer(fh, delimiter='\t')
    header = ['roi_name']
    for task in args.tasks:
        header += [f'{task}_p_raw', f'{task}_p_fdr']
    writer.writerow(header)
    for roi in roi_names_ref:
        row = [roi]
        for task in args.tasks:
            _, p_raw, p_fdr = task_results[task][roi]
            row += [round(float(p_raw), 6), round(float(p_fdr), 6)]
        writer.writerow(row)
log_msg(f"| UPDATE | ROI x task p-value table saved ({len(roi_names_ref)} ROIs x {len(args.tasks)} tasks) → {table_path}")

log_msg(f"| FINISHED | ROI p-values saved → {out_dir}")
