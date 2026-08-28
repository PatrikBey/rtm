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
# Implements Approach 1 from doc/ROISelection.md: null-referenced       #
# significance per ROI. For each task, the observed lvl0 ROI block      #
# z-score (from the SBMFITTING singleflip fit) is compared against the  #
# null distribution of lvl0 z-scores for that ROI (100 behaviour-       #
# permutations, SBMNULL/.../permutation_zscores.tsv), giving a one-     #
# sided empirical p-value: how often chance alone produces a z-score at #
# least as large as the one observed. Benjamini-Hochberg FDR correction #
# is applied per task before thresholding at alpha, exactly as outlined #
# in Approach 1.                                                        #
#                                                                       #
# Outputs:                                                              #
#   roi_pvalues_lvl{level}.tsv   - ROI x task table (raw + FDR p-value  #
#                                   per task)                            #
#   {atlas}_lvl{level}_significant_{task}.nii.gz - one NIfTI per task,  #
#                                   observed z-score kept only for ROIs #
#                                   with FDR p-value < alpha             #
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

args = argparse.ArgumentParser(description='ROI-specific null-referenced p-values (ROISelection.md Approach 1).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+',
                  default=['Foreperiod_Long_tau', 'GoNoGo_tau', 'SATO_Accuracy_tau'],
                  help='Behaviour scores to evaluate (default: all three available tasks)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to test (default: 0)')
args.add_argument('--alpha', type=float, default=0.05, help='FDR-corrected significance threshold (default: 0.05)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMFITTING run directory (default: _singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | ROI p-values (Approach 1) | tasks: {args.tasks} | level {args.level}")


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

    # ---- null distribution: roi x permutation lvl-0 z-scores ---- #
    null_tsv = os.path.join(null_dir, 'permutation_zscores.tsv')
    with open(null_tsv, newline='') as fh:
        reader   = csv.reader(fh, delimiter='\t')
        header   = next(reader)
        null_roi = []
        null_z   = []
        for row in reader:
            null_roi.append(row[0])
            null_z.append([float(v) for v in row[1:]])
    null_roi = np.array(null_roi)
    null_z   = np.array(null_z, dtype=np.float64)   # roi x n_perm
    n_perm   = null_z.shape[1]

    if not np.array_equal(roi_name, null_roi):
        raise ValueError(f"[{task}] ROI order differs between observed fit and null table")

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
